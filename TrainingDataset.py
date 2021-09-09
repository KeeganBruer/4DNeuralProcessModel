from torch.utils.data import Dataset, DataLoader
import glob
import torch
from neural_processes.neural_process import NeuralProcess, NeuralProcessImg
from neural_processes.training import NeuralProcessTrainer
import json
from PIL import Image
import numpy as np
import os
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
        self.data_paths = glob.glob(path_to_data + '/*.npz')
        print(self.data_paths)
        self.file_idx = 0
        self.data = np.load(self.data_paths[self.file_idx])
        self.max_points = max_points if max_points != None else len(self.data_paths) * self.points_per_file
        print("Max Points {}".format(self.max_points))
    def __getitem__(self, index):
        x = []
        y = []
        index = index * self.num_samples
        print("Loading data from index {} to {}".format(index, index+self.num_samples))
        for i in range(self.num_samples):
            new_file_idx = (index+i) // self.points_per_file
            rel_idx = (index+i) % self.points_per_file
            #print("{} {} {}".format(index, new_file_idx, rel_idx))
            if (new_file_idx != self.file_idx):
                self.file_idx = new_file_idx
                self.data = np.load(self.data_paths[self.file_idx])
            x.append(self.data["sample_set_x"][rel_idx])
            y.append([self.data["sample_set_y"][rel_idx]])


        return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))

    def __len__(self):
        return self.max_points // self.num_samples