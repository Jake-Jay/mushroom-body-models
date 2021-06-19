import numpy as np
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset




class MushroomBodyDataset(Dataset):
    def __init__(self, input_dataset, output_dataset, shuffle=False):
        """
        Inputs:
            input_dataset: path to the input dataset
            output_datset: path to the ouput dataset
        """
        
        self.data = self.load_data(input_dataset, output_dataset)

        if shuffle: 
            np.random.shuffle(self.data)

    def load_data(self, input_dataset, output_dataset):
        """
        Inputs:
            data_dir: path to the datasets
            input_dataset: path to the input dataset
            output_datset: path to the ouput dataset
        Outputs:
            data: A list of tuples (DAN time series, MBON time series)
                Each time series has shape (nodes, time steps)
        """

        # Shape of dataset: (15, 10, xxx) = (nodes, time_steps, num_trials)
        X = np.load(input_dataset)
        Y = np.load(output_dataset)

        # return as tuple with input output pairs of shape (timesteps, input_features)
        return [(X[:,:,i].T, Y[:,:,i].T) for i in range(X.shape[2])]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Inputs:
            i: an integer value to index data
        Outputs:
            data: A dictionary of {x:input_time_series, y:output_time_series}
        """
        x, y = self.data[i]
        return {
            'dan': torch.tensor(x).float(),
            'mbon': torch.tensor(y).float()
        }
