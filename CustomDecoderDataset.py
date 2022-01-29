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

    def __init__(self, path_to_data="", max_points = None, device=None):
        self.path_to_data = path_to_data
        self.file = np.load(self.path_to_data)
        self.width = self.file["width"]
        self.height = self.file["height"]
        self.total_frames = self.file["total_frames"]
        self.ppi = self.file["total_frames"]
        self.device = device


    def __getitem__(self, index):
        x = self.file["X"]
        y = self.file["Y"]
        new_y = []
        for y_i in y:
            new_y.append([y_i])
        y = new_y
        return torch.Tensor(np.array(x)).to(self.device), torch.Tensor(np.array(y)).to(self.device)

    def __len__(self):
        return 1
