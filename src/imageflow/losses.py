import torch
import torch.nn.functional as F
import numpy as np

class Grad:
    def __init__(self, penalty="l1", mult=None):
        self.penalty = penalty
        self.mult = mult

    def loss(self, x):
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.mult is not None:
            grad *= self.mult
        return grad


class Rotation:
    def __init__(self, mult=1., derive="central_differences"):
        self.derive = derive
        self.mult = mult
    def loss(self, field):

        if self.derive == "central_differences_exp_scale":
            # f_ = F.pad(field, (1, 1, 0, 0), mode="re")
            # inner_kernel = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
            dx_inner = torch.log(field[:, :, 2:, :]) / torch.log(field[:, :, :-2, :])/2  # F.conv2d(field, inner_kernel)
            dx_0 = torch.log(field[:, :, 1:2, :]) / torch.log(field[:, :, :1, :])
            dx_1 = torch.log(field[:, :, -1:, :]) / torch.log(field[:, :, -2:-1, :])
            dx = torch.cat([dx_0, dx_inner, dx_1], dim=2)

            # print(f"Losses dx: {dx}")

            # inner_kernel = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
            dy_inner = torch.log(field[:, :, :, 2:]) / torch.log(field[:, :, :, :-2])/2  # F.conv2d(field, inner_kernel)
            dy_0 = torch.log(field[:, :, :, 1:2]) / torch.log(field[:, :, :, :1])
            dy_1 = torch.log(field[:, :, :, -1:]) / torch.log(field[:, :, :, -2:-1])
            dy = torch.cat([dy_0, dy_inner, dy_1], dim=3)
            # print(f"Losses dy: {dy}")
            curl = dx[:, 1, ...] / dy[:, 0, ...]
            curl = torch.exp(curl)
            # curl = torch.sum(curl, dim=(1, 2))
            curl = torch.abs(torch.mean(curl))
            # print(f"Losses curl: {curl}")
            return self.mult * curl

        elif self.derive == "central_differences":
            # f_ = F.pad(field, (1, 1, 0, 0), mode="re")
            # inner_kernel = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
            dx_inner = (field[:, :, 2:, :] - field[:, :, :-2, :])/2  # F.conv2d(field, inner_kernel)
            dx_0 = field[:, :, 1:2, :] - field[:, :, :1, :]
            dx_1 = field[:, :, -1:, :] - field[:, :, -2:-1, :]
            dx = torch.cat([dx_0, dx_inner, dx_1], dim=2)

            # print(f"Losses dx: {dx}")

            # inner_kernel = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
            dy_inner = (field[:, :, :, 2:] - field[:, :, :, :-2])/2  # F.conv2d(field, inner_kernel)
            dy_0 = field[:, :, :, 1:2] - field[:, :, :, :1]
            dy_1 = field[:, :, :, -1:] - field[:, :, :, -2:-1]
            dy = torch.cat([dy_0, dy_inner, dy_1], dim=3)
            # print(f"Losses dy: {dy}")
            curl = dx[:, 1, ...] - dy[:, 0, ...]
            # curl = torch.sum(curl, dim=(1, 2))
            curl = torch.abs(torch.mean(curl))
            # print(f"Losses curl: {curl}")
            return self.mult * curl

        else:
            raise NotImplemented("No available differentiation method.")



