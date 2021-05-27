import numpy as np
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset




class MushroomBodyDataset(Dataset):
    def __init__(self, data_dir:str = '../data', shuffle=True):
        """
        Inputs:
            data: list of tuples (input time series, output time series)
        """
        
        self.data = self.load_data(data_dir)

        if shuffle: 
            np.random.shuffle(self.data)

    def load_data(self, data_dir):
        """
        Outputs:
            data: A list of tuples (DAN time series, MBON time series)
                Each time series has shape (nodes, time steps)
        """
        
        DATA_DIR = Path(data_dir)

        # Shape of dataset: (15, 3, 9000) = (nodes, time_steps, trials)
        X = np.load(DATA_DIR / 'X-time-series-from-distribution.npy')
        Y = np.load(DATA_DIR / 'Y-time-series-from-distribution.npy')

        input_features, timesteps, num_trials = X.shape

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
