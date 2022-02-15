import glob
import torch
from neural_processes.neural_process import NeuralProcess, NeuralProcessImg
from neural_processes.training import NeuralProcessTrainer
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
import os
from CustomJsonEncoder import CustomJsonEncoder
from CustomDecoderDataset import TestData

import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(config):
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    r_dim = config["r_dim"]
    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    num_context_range = config["num_context_range"]
    num_extra_target_range = config["num_extra_target_range"]
    epochs = config["epochs"]
    save_directory = config["save_directory"]
    data_directory = config["data_directory"]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    print(config)
    # Define neural process for functions...
    neuralprocess = NeuralProcess(x_dim=7, y_dim=1, r_dim=12, z_dim=12, h_dim=4).to(device)
    neuralprocess.training = True


    # Define optimizer and trainer
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-10)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=num_context_range,
                                      num_extra_target_range=num_extra_target_range, print_freq=1)

    dataset = TestData(path_to_data=data_directory, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("starting training")
    # Train on your data
    epoch_history = []
    for epoch in range(epochs):
        print("Epoch {0:3d}:".format(epoch + 1))
        if not os.path.exists(save_directory + "/epoch{}".format(epoch+1)):
            os.makedirs(save_directory + "/epoch{}".format(epoch+1))
        loss = np_trainer.train(data_loader, save_directory=save_directory+"/epoch{}".format(epoch+1))
        epoch_history.append(loss)
        print("Epoch {0:3d}, Loss: {1}".format(epoch + 1, loss))
        # Save losses at every epoch
        with open(save_directory + '/losses.json', 'w') as f:
            json.dump(epoch_history, f, indent=4, cls=CustomJsonEncoder)
        # Save model at every epoch
        torch.save(np_trainer.neural_process.state_dict(), save_directory + '/model.pt')
        torch.cuda.empty_cache()



if __name__ == "__main__":
    config_path = input("config path (Press enter for default)>")
    config = {}
    with open(config_path if config_path!="" else "./config.json") as config_file:
        config = json.load(config_file )
    run_training(config)
