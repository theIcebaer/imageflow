import torch
import torch.nn as nn

from iunets import iUNet
from voxelmorph.torch.layers import VecInt, SpatialTransformer
from imageflow.dataset import MnistDataset


class FlowIUnet(nn.Module):
    def __init__(self, data_res=(28, 28), **kwargs):
        super().__init__()
        if kwargs.get['device']:
            self.device = kwargs.get['device']
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.iunet = iUNet(channels=(2, 8, 16, 32, 40), architecture=(2, 2, 2, 2), dim=2)
        self.integrator = VecInt(inshape=data_res, nsteps=7)
        self.transformer = SpatialTransformer(size=data_res).to(self.device)

    def forward(self, x):
        m = x[:, 0, ...]
        f = x[:, 1, ...]
        v = self.iunet(x)
        t = self.integrator(v)
        x = self.transformer(m, t)
        return x


