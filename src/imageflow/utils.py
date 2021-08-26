import torch
import numpy as np

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

