import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageflow
from imageflow.nets import Reg_mnist_cINN
from imageflow.nets import CinnBasic
from imageflow.nets import CinnConvMultiRes
from imageflow.dataset import MnistDataset, BirlData, FireDataset
from imageflow.utils import apply_flow
from imageflow.dataIO import load_dataset
from imageflow.modelIO import load_model
from imageflow.testDistribution import _get_samples, _visualize_embedding

from torch.nn.functional import mse_loss as mse
from torch.utils.data import DataLoader
import math

def test_model(data="MNIST", file_name="mnist_rnd_distortions_1.hdf5", cond_net=None, plot=True, plot_dir='plots',
               batch_size=256, quiver=False, stream=True, figs=True, data_res=(28, 28), n_plot_samples=1, **config):
    # data dir
    # model dir
    # model type
    # test batch size
    # plot
    # figure_dir
    ndim_total = 28 * 28 * 2

    if plot:
        print(plot_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
    else:
        plot_dir = None



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_dir = 'data'
    # print(config.get("data_dir"))
    # print(config.get("model_dir"))
    if config.get("data_dir"):
        # print(config.get("data_dir"))
        data_dir = config.get("data_dir")

    if data == 'MNIST':
        print("Initializing MNIST data.")
        # print(data_dir)
        data_set = MnistDataset(os.path.join(data_dir, file_name), noise=0.08)  # also try 0.005
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        test_indices = indices[44000:]

        if config.get("index") is not None:
            index = config.get("index")
            test_indices = test_indices[index]


        test_set = torch.utils.data.Subset(data_set, test_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))

    elif data == "Birl":
        print("Initializing Birl dataset")
        test_set = BirlData(os.path.join(data_dir, 'birl'), mode='test', color='grayscale', sample_res=data_res)
        # val_set = BirlData(os.path.join(data_dir, 'birl'), mode='validation', color='grayscale', sample_res=data_res)
    elif data == "FIRE":
        size = config.get("image_res")[0] if config.get("image_res") else 182
        data_set = FireDataset(data_dir, size)
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        test_indices = indices[108:]
        test_set = torch.utils.data.Subset(data_set, test_indices)

    elif data=="Covid":
        from imageflow.dataset import CovidXDataset
        data_set = CovidXDataset(data_dir)
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        n_test = int(n_samples * 0.8)
        test_indices = indices[n_test:]
        # val_indices = indices[n_train: n_train+n_val]

        test_set = torch.utils.data.Subset(data_set, test_indices)


    else:
        raise KeyError("No valid dataset configured. Use data='MNIST' or data='Birl'.")

    # if data == "Birl":
    #     shuffle = False
    # else:
    #     shuffle = True
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)


    if config.get("model_dir"):
        from imageflow.utils import load_checkpoint
        print("Loading model from file...")
        model_dir = config.get("model_dir")
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
                cinn = CinnBasic(cond_net=cond_net, **model_architecture)

            elif model_type == "multiRes":
                args = [arg for arg in ["block_depth", "cond_channels", "splits", "downsample", "conv_channels"] if
                        arg in config.keys()]
                image_res = config.get("image_res") if config.get("image_res") else None
                model_architecture = {key: config[key] for key in args}
                cinn = CinnConvMultiRes(data_res=data_res, image_res=image_res, **model_architecture)
            else:
                raise KeyError("No model known type specified, specify: model_type=basic or model_type=multiRes.")
        else:
            raise KeyError("No model type specified, specify: model_type=basic or model_type=multiRes.")
        # Model = getattr(sys.modules[imageflow.nets], model_name)
        # cinn = CinnConvMultiRes(device=device)
        cinn.to(device)

        state_dict = {k: v for k, v in model_dict['state_dict'].items() if 'tmp_var' not in k}
        cinn.load_state_dict(state_dict)
        print("...done")

    elif config.get("model_object"):
        cinn = config.get("model_object")
    else:
        raise("No model specified")

    cinn.eval()


    with torch.no_grad():
        field_err = []
        reconstruction_err = []


        for i, batch in enumerate(test_loader):
            if data == "Birl" or data=="FIRE" or data=="Covid":
                c = batch
                if i > batch_size: break
            else:
                v_field, c = batch
                v_field = v_field.to(device)
            # print(v_field.shape)
            c = c.to(device)
            # x = true field
            # c = condition vector of composed moving and fixed image (m,f)
            for k_sample in range(n_plot_samples):

                z = torch.randn(batch_size, 2 * math.prod(data_res)).to(device)
                if config.get("latent_zero"):
                    z = torch.zeros((batch_size, 2 * math.prod(data_res))).to(device)
                target_pred, v_field_pred, _ = cinn.reverse_sample(z, c=c)  # predicted velocity field

                source_imgs = c[:, 0, ...]

                source = c[:, :1, ...].to(device)
                target = c[:, 1:, ...].to(device)

                target_pred, _ = apply_flow(v_field=v_field_pred, volume=source, device=device)

                if data == "MNIST":
                    v_field = v_field.to(device)
                    field_err.extend(np.mean(np.square(v_field.cpu().detach().numpy() - v_field_pred.cpu().detach().numpy()), axis=(1, 2, 3)))
                # reconstruction_err.append(mse(target, target_pred).item())
                reconstruction_err.extend(np.mean(np.square(target.cpu().detach().numpy() - target_pred.cpu().detach().numpy()), axis=(1, 2, 3)))
                if plot:
                    # Shift everything to cpu since you can't plot from gpu directly.
                    source = source.to(torch.device("cpu"))
                    target = target.to(torch.device("cpu"))
                    target_pred = target_pred.to(torch.device("cpu"))
                    v_field_pred = v_field_pred.to(torch.device("cpu"))
                    if data == "MNIST":
                        v_field = v_field.to(torch.device("cpu"))

                    from imageflow.visualize import streamplot_from_batch
                    from imageflow.visualize import make_fig_from_batch
                    from imageflow.visualize import quiver_from_batch

                    path1 = os.path.join(plot_dir, "true_fields")
                    path2 = os.path.join(plot_dir, "pred_fields")
                    path3 = os.path.join(plot_dir, "true_field_quiver")
                    path4 = os.path.join(plot_dir, "pred_field_quiver")
                    # print(path1)
                    # print(path2)
                    if n_plot_samples == 1:
                        k_sample=None

                    if data == "MNIST":
                        if stream:
                            streamplot_from_batch(v_field, show_image=False, get_all=True, save_path=path1, affix=k_sample)
                        if quiver:
                            quiver_from_batch(v_field, show_image=False, get_all=True, save_path=path3, affix=k_sample)
                    if stream:
                        streamplot_from_batch(v_field_pred, show_image=False, get_all=True, save_path=path2, affix=k_sample)
                    if quiver:
                        quiver_from_batch(v_field_pred, show_image=False, get_all=True, save_path=path4, affix=k_sample)

                    if figs:
                        make_fig_from_batch(source, save_path=os.path.join(plot_dir, "source"), affix=k_sample)
                        make_fig_from_batch(target, save_path=os.path.join(plot_dir, "target"), affix=k_sample)
                        make_fig_from_batch(target_pred, save_path=os.path.join(plot_dir, "predicted"), affix=k_sample)
                    if config.get("latent_zero"):
                        break

            plot = False
            if config.get('only_plot'):
                return reconstruction_err, field_err

        return reconstruction_err, field_err


def make_embeddings(
        model_dir,
        idx=(0,),
        data_res=(28, 28),
        data="MNIST",
        data_dir="/home/jens/thesis/imageflow/data",
        file_name="mnist_rnd_distortions_1.hdf5",
        method='umap',
        **config):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_set = load_dataset(data="MNIST", **config)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    cinn = load_model(model_dir, data_res, **config)
    cinn.to(device)
    cinn.eval()

    for i in idx:
        field, cond = test_set[i]
        field, cond = field.unsqueeze(dim=0).to(device), cond.unsqueeze(dim=0).to(device)

        source, target = cond[:, :1, ...], cond[:, 1:, ...]
        # print(source.shape)
        # print(target.shape)
        # z = torch.
        # p_field, p_target = cinn(cond)
        field_samples, pred_samples = _get_samples(cond, cinn, device, n_samples=500)

        if config.get("show_sample"):
            from imageflow.visualize import streamplot_from_batch
            streamplot_from_batch(field_samples[0][:10].cpu().detach().numpy(), show_image=True, get_all=True)


        _visualize_embedding(field_samples[0], method=method)