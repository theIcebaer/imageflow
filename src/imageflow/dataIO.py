import torch
import os
import numpy as np
from imageflow.nets import CinnBasic, CinnConvMultiRes
from imageflow.dataset import MnistDataset, BirlData, FireDataset

def load_dataset(data, **config):

    if data == 'MNIST':
        print("Initializing MNIST data.")
        # print(data_dir)

        data_dir = "/home/jens/thesis/imageflow/data" if not config.get("data_dir") else config.get("data_dir")
        file_name = "mnist_rnd_distortions_1.hdf5" if not config.get("file_name") else config.get("file_name")

        data_set = MnistDataset(os.path.join(data_dir, file_name), noise=0.08)  # also try 0.005
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        test_indices = indices[44000:]

        if config.get("index") is not None:
            index = config.get("index")
            test_indices = test_indices[index]


        test_set = torch.utils.data.Subset(data_set, test_indices)


    elif data == "Birl":
        print("Initializing Birl dataset")
        data_res = config.get("data_res") if config.get("data_res") else (64, 64)
        data_dir = "/home/jens/thesis/imageflow/data" if not config.get("data_dir") else config.get("data_dir")
        test_set = BirlData(os.path.join(data_dir, 'birl'), mode='test', color='grayscale', sample_res=data_res)

    elif data == "FIRE":

        data_dir = "/home/jens/thesis/imageflow/data" if not config.get("data_dir") else config.get("data_dir")
        data_set = FireDataset(data_dir)
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        test_indices = indices[108:]
        test_set = torch.utils.data.Subset(data_set, test_indices)
    else:
        raise AttributeError("No valid dataset configured. Use data='MNIST', data='Birl', data=FIRE.")

    return test_set