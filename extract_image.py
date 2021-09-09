from neural_processes.neural_process import NeuralProcess, NeuralProcessImg
from neural_processes.utils import (context_target_split, batch_context_target_mask,
                   img_mask_to_np_input)
import glob
import torch
from random import randint
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
import os
from CustomJsonEncoder import CustomJsonEncoder
from TrainingDataset import TestData
torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageExtractor():
    def __init__(self, config):
        self.img_size = config["img_size"]
        batch_size = config["batch_size"]
        r_dim = config["r_dim"]
        h_dim = config["h_dim"]
        z_dim = config["z_dim"]
        self.num_context_range = config["num_context_range"]
        self.num_extra_target_range = config["num_extra_target_range"]
        epochs = config["epochs"]
        self.save_directory = config["save_directory"]
        self.data_directory = config["data_directory"]
        self.neural_process = NeuralProcess(x_dim=3, y_dim=3, r_dim=r_dim, z_dim=z_dim, h_dim=h_dim).to(device)
        self.neural_process.load_state_dict(torch.load(self.save_directory + '/model.pt', map_location=lambda storage, loc: storage))
        self.neural_process.training = False
        dataset = TestData(num_samples=10, num_points=2, path_to_data=self.data_directory, device=device)
        self.dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def extract_images(self):
        for i, data in enumerate(self.dataset):
            x, y = data  # data is a tuple (img, label)
            x_context = x.to(device)
            y_context = y.to(device)
            num_context = randint(*self.num_context_range)
            num_extra_target = randint(*self.num_extra_target_range)
            x_target = []
            width, height, time = 10, 10, 10
            for i in range(time):
                for x in range(width):
                    for y in range(height):
                        x_target.append([(x/(width/2))-1, (y/(height/2))-1, i/time])
            x_target = [x_target]
            x_target = torch.Tensor(np.array(x_target))
            p_y_pred = self.neural_process(x_context, y_context, x_target)
            print(p_y_pred.loc.detach().cpu())

if __name__ == "__main__":
    config_path = input("config path>")
    config = {}
    with open(config_path) as config_file:
        config = json.load(config_file)
    ImgExtractor = ImageExtractor(config)
    ImgExtractor.extract_images()