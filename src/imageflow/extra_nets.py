import math
import os
import torch
import torch.nn as nn
import numpy as np

# import FrEIA.framework as Ff
# import FrEIA.modules as Fm

# os.environ['VXM_BACKEND'] = 'pytorch'
# from voxelmorph.torch.layers import VecInt
# from voxelmorph.torch.layers import SpatialTransformer
#
# from torchvision.models import mobilenet_v3_large
# from torchvision.models import resnet18
# from imageflow.inn_builder import build_inn


class ScalingCondNet(nn.Module):

    def __init__(self, data_res=(28, 28), channels=(64, 128, 128)):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        def _get_lin_res(data_res, kernel_size):
            h_in = np.array([x/ (2**(len(channels)-1)) for x in data_res])
            h_out = np.floor((h_in-kernel_size)/kernel_size + 1)
            lin_shape = int(channels[-1] * h_out[0] * h_out[1])
            return lin_shape

        avg_pool_kernel_size = 3 if data_res[0] < 64 else 4

        lin_shape = _get_lin_res(data_res, avg_pool_kernel_size)

        # self.resolution_levels = nn.ModuleList([
        #     nn.Sequential(nn.Conv2d(2, channels[0], 3, padding=1), nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1)),
        #     nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.LeakyReLU(),
        #                   nn.Conv2d(channels[1], channels[1], 3, padding=1, stride=2)),
        #     nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2)),
        #     nn.Sequential(nn.LeakyReLU(), nn.AvgPool2d(kernel_size=avg_pool_kernel_size), Flatten(), nn.Linear(lin_shape, 512))
        #     ])
        self.resolution_levels = nn.ModuleList([nn.Sequential(nn.Conv2d(2, channels[0], 3, padding=1), nn.LeakyReLU(),
                                                              nn.AdaptiveAvgPool2d(output_size=data_res[0]),
                                                              nn.Conv2d(channels[0], channels[0], 3, padding=1))])
        for i in range(len(channels)-2):
            self.resolution_levels.append(
                nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[i], channels[i+1], 3, padding=1), nn.LeakyReLU(),
                              # nn.AdaptiveAvgPool2d(output_size=out_shape[i+1]),
                              nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1, stride=2)))
        self.resolution_levels.append(nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[-2], channels[-1], 3, padding=1, stride=2)))
        self.resolution_levels.append(nn.Sequential(nn.LeakyReLU(), nn.AvgPool2d(kernel_size=avg_pool_kernel_size), Flatten(), nn.Linear(lin_shape, 512)))

    def forward(self, c):
        outputs = [c]
        for i, m in enumerate(self.resolution_levels):
            outputs.append(m(outputs[-1]))
        return outputs[1:]