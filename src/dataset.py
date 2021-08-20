import torch
from pathlib import Path
import h5py
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    """
    This dataset assumes the data is given as hdf5 dataset in the following format:
    /source/samples
    /target/samples
    /v_field/samples
    /deformations/samples
    """

    def __init__(self, file_path, access_mode='r'):
        file_path = Path(file_path)
        self.file = h5py.File(file_path, access_mode)

    def __len__(self):
        return len(self.file["source"])

    def __getitem__(self, idx):
        v_field = torch.tensor(self.file['v_field'][idx])
        source = torch.tensor(self.file['source'][idx])
        target = torch.tensor(self.file['target'][idx])
        condition = torch.cat((source, target), dim=0)
        return v_field, condition

