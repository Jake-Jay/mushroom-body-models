import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import pdb

from masked_linear import CustomizedLinear


class NeuralRNNModule(pl.LightningModule):
    def __init__(self, masked=False, lr=0.001, **kwargs):
        """
        Inputs:
            - input_dim: shape of input time series
            - hidden_dim: hidden_size of the rnn layer
            - lr: learning rate
        """
        super().__init__()

        # Hyperparameters
        self.lr = lr
        non_linearity = kwargs['non_linearity'] if 'non_linearity' in kwargs else 'tanh'
        self.save_hyperparameters('lr', 'non_linearity', 'masked')

        if masked:
            self.input_mask=kwargs['input_mask']   # (15,15)
            self.hidden_mask=kwargs['hidden_mask'] # (15,15)

            # Define model using masked RNN
            self.rnn = MaskedNeuralRNN(
                input_mask=self.input_mask,
                hidden_mask=self.hidden_mask,
                nonlinearity=non_linearity,
            )
        else:
            self.input_dim = kwargs['input_dim'] if 'input_dim' in kwargs else 15
            self.hidden_dim = kwargs['hidden_dim'] if 'hidden_dim' in kwargs else 15

            # Define model using custom RNN
            self.rnn = NeuralRNN(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                nonlinearity=non_linearity,
            )

        # Define loss function
        self.loss_fcn = nn.MSELoss()

    def forward(self, sequence):
        """
        Inputs
            sequence: A long tensor of Size ([8, 3, 15])
        Outputs:
            output: A long tensor of Size ([8, 3, 15])
        """

        rnn_out, _ = self.rnn(sequence)
        return rnn_out
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams({"hp/lr": self.lr})

    def training_step(self, batch, batch_idx):
        inputs = batch['dan']     # (neurons, timesteps, batchsize)
        outputs = batch['mbon']    # (neurons, timesteps, batchsize)

        preds = self(inputs)        # (neurons, timesteps, batchsize)
        loss = self.loss_fcn(preds, outputs)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['dan']      # (neurons, timesteps, batchsize)
        outputs = batch['mbon']    # (neurons, timesteps, batchsize)

        preds = self(inputs)        # (neurons, timesteps, batchsize)
        loss = self.loss_fcn(preds, outputs)

        self.log('val/loss', loss)
        self.log("hp/lr", self.lr)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }

class NeuralRNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=15, nonlinearity="tanh"):
        super().__init__()

        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_xh = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.W_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        if nonlinearity == "tanh":
            self.activation = nn.Tanh() 
        elif nonlinearity == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unrecognized activation. Allowed activations: tanh or relu")

    def forward(self, x):
        """
        Inputs:
        - x: Input tensor (batch_size, seq_len, input_size)

        Outputs:
        - h_seq: Hidden vector along sequence (batch_size, seq_len, input_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """

        # TODO implement weights matrix that will remove certain recurrent connections

        # Hidden State initialization
        h = torch.zeros(
            (1, x.size(0), self.hidden_size), 
            device=x.device, 
            dtype=x.dtype
        )                                   # (1, batch, hidden_features)

        h_seq = []
        for xt in x.unbind(1):
            # update hidden state
            h = self.activation(self.W_hh(h) + self.W_xh(xt))
            h_seq.append(h)

        # Stack the h_seq list as a tensor along dim 0
        h_seq = torch.cat(h_seq, 0)         # (seq_len, batch, hidden_features)

        # Batch first
        h_seq = h_seq.permute(1,0,2)        # (batch, seq_len, hidden_features)

        return h_seq, h

class MaskedNeuralRNN(nn.Module):
    '''Applies modified linear layer at each time step to constrain the optimisation of weights'''
    def __init__(self, input_mask, hidden_mask, nonlinearity="tanh"):
        super().__init__()

        """
        Inputs:
        - input_mask: mask to constrain Wxh (input_features, hidden_features)
        - hidden_mask: mask to contrain Whh (hidden_features, output_features)
        - activation: Nonlinearity in cell; 'tanh' or 'relu'

        See about masking here:
        https://stackoverflow.com/questions/53544901/how-to-mask-weights-in-pytorch-weight-parameters
        """
        self.input_mask = input_mask
        self.hidden_mask = hidden_mask
        self.hidden_size = hidden_mask.shape[0]

        self.W_xh = CustomizedLinear(self.input_mask, bias=True)
        self.W_hh = CustomizedLinear(self.hidden_mask, bias=True)

        if nonlinearity == "tanh":
            self.activation = nn.Tanh() 
        elif nonlinearity == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unrecognized activation. Allowed activations: tanh or relu")
    
    def forward(self, x):
        """
        Inputs:
        - x: Input tensor (batch_size, seq_len, input_size)

        Outputs:
        - h_seq: Hidden vector along sequence (batch_size, seq_len, input_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """

        # Hidden State initialization
        h = torch.zeros(
            (x.size(0), self.hidden_size), 
            device=x.device, 
            dtype=x.dtype
        )                                   # (batch, hidden_features)

        h_seq = []
        for xt in x.unbind(1):
            # update hidden state
            h = self.activation(self.W_hh(h) + self.W_xh(xt))
            h_seq.append(h.expand(1 ,-1, -1))

        # Stack the h_seq list as a tensor along dim 0
        h_seq = torch.cat(h_seq, 0)         # (seq_len, batch, hidden_features)

        # Batch first
        h_seq = h_seq.permute(1,0,2)        # (batch, seq_len, hidden_features)

        return h_seq, h




