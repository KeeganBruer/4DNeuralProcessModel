from torch.utils.data import Dataset, DataLoader
import glob
import torch
from neural_processes.neural_process import NeuralProcess, NeuralProcessImg
from neural_processes.training import NeuralProcessTrainer
import json
from PIL import Image
import numpy as np
import os
import random
class TestData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------

    num_samples : int
        Number of samples of the function contained in dataset.
    """

    def __init__(self, num_samples=10, points_per_file=10, path_to_data="", max_points = None, device=None):
        self.num_samples = num_samples
        self.points_per_file = points_per_file
        self.data_paths = sorted(glob.glob(path_to_data + '/*.npz'), key=lambda x: int(os.path.basename(x).replace(".npz", "").split('_')[-1]))
        self.device = device
        print(self.data_paths)
        self.max_points = max_points if max_points != None else len(self.data_paths) * self.points_per_file
        print("Max Points {}".format(self.max_points))


    def __getitem__(self, index):
        x = []
        y = []
        print(index)

        start_frame = np.load(self.data_paths[index])
        target_frame = 0
        end_frame = np.load(self.data_paths[index+2])
        for i in range(self.num_samples):
            ridx1 = round(random.random() * len(start_frame))
            ridx2 = round(random.random() * len(end_frame))
            
            x.append(start_frame["sample_set_x"][ridx1])
            tmp_y = [end_frame["sample_set_y"][ridx1]]
            y.append(tmp_y)

            x.append(end_frame["sample_set_x"][ridx2])
            tmp_y = [end_frame["sample_set_y"][ridx2]]
            #print(tmp_y) 
            y.append(tmp_y)
        
        return torch.Tensor(np.array(x)).to(self.device), torch.Tensor(np.array(y)).to(self.device)

    def __len__(self):
        return len(self.data_paths)-2
