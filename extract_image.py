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
import math
from CustomJsonEncoder import CustomJsonEncoder
from TrainingDataset import TestData
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
        self.neural_process = NeuralProcess(x_dim=8, y_dim=1, r_dim=r_dim, z_dim=z_dim, h_dim=h_dim).to(device)
        self.neural_process.load_state_dict(torch.load(self.save_directory + '/model.pt', map_location=lambda storage, loc: storage))
        self.neural_process.training = False
        dataset = TestData(num_samples=500, points_per_file=10000, max_points=3000, path_to_data=self.data_directory, device=device)
        self.dataset = DataLoader(dataset, batch_size=1, shuffle=True)

    def extract_images(self):
        for i, data in enumerate(self.dataset):
            x, y = data  # data is a tuple (img, label)
            x_context = x.to(device)
            y_context = y.to(device)
            x_target = []
            width, height, time = 10, 10, 30
            view_distance = 1000
            origin = [0, -3, 0]

            for i in range(time):
                for x in range(width):
                    for y in range(height):
                        z = math.sqrt(view_distance ** 2 - x ** 2 - y ** 2)
                        new_point = [origin[0] + x, origin[1] + y, origin[2] + z]
                        new_x = [*origin, time, *new_point, time]
                        x_target.append(new_x)
            x_target = [x_target]
            print(np.array(x_target).shape)
            print(np.array(x_context.to(torch.device("cpu"))).shape)
            x_target = torch.Tensor(np.array(x_target)).to(device)
            p_y_pred = self.neural_process(x_context, y_context, x_target)

            y_target = p_y_pred.loc.detach().cpu()
            print(x_target[0][0])
            print(y_target[0][0])

if __name__ == "__main__":
    config_path = input("config path (Press enter for default)>")
    config = {}
    with open(config_path if config_path != "" else "./config.json") as config_file:
        config = json.load(config_file)
    ImgExtractor = ImageExtractor(config)
    ImgExtractor.extract_images()