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
from CustomDecoderDataset import TestData
from Dataset_converter import convert_dataset

import plotly.express as px
import plotly.graph_objects as go
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
        self.neural_process = NeuralProcess(x_dim=7, y_dim=1, r_dim=100, z_dim=12, h_dim=4).to(device)
        self.neural_process.load_state_dict(torch.load(self.save_directory + '/model.pt', map_location=lambda storage, loc: storage))
        self.neural_process.training = False
        self.dataset_ref = TestData(path_to_data=self.data_directory, device=device)
        self.dataset = DataLoader(self.dataset_ref, batch_size=1, shuffle=True)

        if not os.path.exists("images"):
            os.mkdir("images")

    def extract_images(self):
        width, height, time = 10, 10, 0.1
        for i, data in enumerate(self.dataset):
            x, y = data  # data is a tuple (img, label)
            x_context = x.to(device)
            y_context = y.to(device)
            #print(x_context)
            for batch_i in range(0, len(x_context.tolist())):
                batch = x_context.tolist()[batch_i]
                #y_batch = y_context.tolist()[batch_i]
                #for ray_i in range(0, len(batch)):
                #    print(batch[ray_i], y_batch[ray_i])
                x_target = x_context
                p_y_pred = self.neural_process(x_context, y_context, x_target)
                for ray_i in range(0, len(batch)):
                    print(batch[ray_i], p_y_pred[ray_i])
                '''
                np.savez_compressed(
                    "extracted_rays.npz", 
                    X=batch, Y=y_batch, 
                    width=self.dataset_ref.width, 
                    height=self.dataset_ref.height,
                    total_frames=self.dataset_ref.total_frames,
                    ppi=self.dataset_ref.ppi
                )
                '''
            return
            points.append(origin)
            for point in points:
                x_target.append([*origin, time, *point, time])
            x_target = [x_target]
            print(np.array(x_context.to(torch.device("cpu"))).shape)
            print(np.array(x_target).shape)
            x_target = torch.Tensor(np.array(x_target)).to(device)
            p_y_pred = self.neural_process(x_context, y_context, x_target)
            
            x_target = x_target.cpu().numpy()
            y_target = p_y_pred.loc.detach().cpu().numpy()
            print(x_target)

if __name__ == "__main__":
    config_path = input("config path (Press enter for default)>")
    config = {}
    with open(config_path if config_path != "" else "./config.json") as config_file:
        config = json.load(config_file)
    ImgExtractor = ImageExtractor(config)
    ImgExtractor.extract_images()
