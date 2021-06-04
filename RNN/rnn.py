import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import pdb


class NeuralRNNModule(pl.LightningModule):
    def __init__(self, input_dim=15, hidden_dim=15, lr=0.001, **additional_kwargs):
        """
        Inputs:
            - input_dim: shape of input time series
            - hidden_dim: hidden_size of the rnn layer
            - lr: learning rate
        """
        super().__init__()

        # Hyperparameters
        self.lr = lr
        self.save_hyperparameters('lr')

        # Define model using custom RNN
        self.rnn = NeuralRNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            nonlinearity='tanh',
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
            optimizer, patience=5, verbose=True)
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

        See about masking here:
        https://stackoverflow.com/questions/53544901/how-to-mask-weights-in-pytorch-weight-parameters
        """

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_xh = nn.Linear(self.input_size, self.hidden_size, bias=True)
        # self.M_xh = torch.ones(self.input_size, self.hidden_size)   Mask
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

    




"""
extended torch.nn module with sparse connectivity.
This code base on https://pytorch.org/docs/stable/notes/extending.html

Author's repository:
https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
"""
import math

#################################
# Define custome autograd function for masked connection.

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module with mask connection.
        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )



