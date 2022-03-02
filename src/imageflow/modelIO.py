import torch
from imageflow.nets import CinnBasic, CinnConvMultiRes
from imageflow.utils import load_checkpoint

def load_model(model_dir, data_res=(28,28), **config):

    print("Loading model from file...")
    model_dict = torch.load(model_dir)
    if model_dict.get("model_architecture"):
        cinn, _, _ = load_checkpoint(model_dict)
    elif config.get("model_type"):
        model_type = config.get('model_type')
        if config.get("model_type") == "basic":
            args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            # print(f"Creating basic cinn with conditioning network {type(cond_net)}")
            # Model = getattr(sys.modules[imageflow.nets], )
            cond_net = config.get("cond_net")
            cinn = CinnBasic(cond_net=cond_net, **model_architecture)

        elif model_type == "multiRes":
            args = [arg for arg in ["block_depth", "cond_channels", "splits", "downsample", "conv_channels"] if
                    arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            cinn = CinnConvMultiRes(data_res=data_res, **model_architecture)
        else:
            raise KeyError("No model known type specified, specify: model_type=basic or model_type=multiRes.")
    else:
        raise KeyError("No model type specified, specify: model_type=basic or model_type=multiRes.")

    state_dict = {k: v for k, v in model_dict['state_dict'].items() if 'tmp_var' not in k}
    cinn.load_state_dict(state_dict)
    return cinn