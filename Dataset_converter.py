import numpy as np

def convert_dataset(file_path):
    data = np.load(file_path)
    print(data.files)
    print(data["sample_set_x"])
    print(data["sample_set_y"])


if __name__ == "__main__":
    convert_dataset("./training_data/sample_set_0.npz")
