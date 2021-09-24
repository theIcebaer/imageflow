import os
import torch
import numpy as np
import subprocess
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch.layers import VecInt, SpatialTransformer

import imageflow as imf

def apply_flow(volume, v_field, device=torch.device('cpu'), integrator=None, transformer=None):
    """This function just applies a flow field to some data, but it is honestly a bit obsulete, especially if one
    defines integrator and transformer outside, which is needed if those are part of a network. It is only useful in a
    test environment.
    """
    # Convert to torch tensor if needed.
    if type(volume) == np.ndarray:
        volume = torch.from_numpy(volume)
    if type(v_field) == np.ndarray:
        v_field = torch.from_numpy(v_field)
    # Define local integrator and transformer if no external one is given.
    if integrator is None:
        integrator = VecInt(inshape=v_field.shape[2:], nsteps=7).to(device)
    if transformer is None:
        transformer = SpatialTransformer(size=volume.shape[2:]).to(device)


    deformation = integrator(v_field)
    transformed = transformer(volume, deformation)

    return transformed, deformation


def make_models(conf):
    """ building list of networks to train from a config dict. Maybe do a yaml file or something else for this later.
    config options:
    - supervised / unsupervised
    - conditioning network
        - pretrained or not
    - cINN structure
        - plain/multiresolution
        -
    """

    conf = {
        "supervised": True,
        "cond_net": "resnet",
        "pretrained": True,
        "cINN": "basic"
    }

    if conf.get("supervised"):
        if conf.get("cINN") == "basic":
            if conf.get("cond_net") is not None:
                if conf.get("cond_net") == "resnet":
                    from imageflow.nets import CondNet
                    from torchvision.models import resnet18

                    ext_model = resnet18(pretrained=conf.get("pretrained"))
                    cond_net = imf.nets.CondNetWrapper(ext_model, type='resnet')
                    cinn = imf.nets.CinnBasic(cond_net=cond_net)

        elif conf.get("cINN") == "convolutional":
            from imageflow.nets import CinnConvMultires

            cond_net = CinnConvMultires()


    else:  # unsupervised
        pass


def get_commit():
    """Little helper script track code version per run and provide a link to the github view of the respective commit
    to ease the workflow.
    """
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('ascii').strip()
    link = f"https://github.com/theIcebaer/imageflow/tree/{commit}"
    return commit, link


# def get_cond_shape(cond_net, data_shape):
#     """Helper function to infer output shape of a conditioning network from the network type. Not very elegant, but
#     better than hardcoding it inside of the class definition.
#     """
#
#     if type(cond_net) ==
#
#     def init_weights(self, method="gaussian", init_args=None):
#         if method == "gaussian":
#             for p in self.cinn.parameters():
#                 if p.requires_grad:
#                     if init_args is None:
#                         p.data = 0.01 * torch.randn_like(p)
#                     else:
#                         p.data = init_args['lr'] * torch.randn_like(p)
#
#         elif method == "xavier":
#             for p in self.cinn.parameters():
#                 if p.requires_grad:
#                     if init_args is None:
#                         torch.nn.init.xavier_uniform_(p, gain=1.0)
#                     else:
#                         torch.nn.init.xavier_uniform_(p, gain=init_args['gain'])