import os
import sys

import pandas
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import imageflow
from imageflow.nets import Reg_mnist_cINN
from imageflow.nets import CinnBasic
from imageflow.nets import CinnConvMultiRes
from imageflow.dataset import MnistDataset, BirlData
from imageflow.utils import apply_flow
from torch.nn.functional import mse_loss as mse
from torch.utils.data import DataLoader
import math
import seaborn as sns

import itertools

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

def get_err_distribution(data="MNIST", file_name="mnist_rnd_distortions_1.hdf5", cond_net=None, plot=True, plot_dir='plots',
               batch_size=10, data_res=(28, 28), samples_per_datapoint=100, **config):

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
    else:
        raise KeyError("No valid dataset configured. Use data='MNIST' or data='Birl'.")

    # if data == "Birl":
    #     shuffle = False
    # else:
    #     shuffle = True
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)


    if config.get("model_dir"):
        print("Loading model from file...")
        model_dir = config.get("model_dir")
        model_dict = torch.load(model_dir)
        if config.get("model_type"):
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
                model_architecture = {key: config[key] for key in args}
                cinn = CinnConvMultiRes(data_res=data_res, **model_architecture)
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
        field = {
            'error': [],
            'mean': [],
            'std': []
        }

        v_field, cond = next(iter(test_loader))
        v_field = v_field.to(device)
        cond = cond.to(device)

        fields, _ = _get_samples(cond, cinn, device, n_samples=samples_per_datapoint)


        for k in range(batch_size):
            errors = [mse(v_field[k], f).cpu().detach().numpy().item() for f in fields[k]]
            std = np.std(errors)
            mean = np.mean(errors)

            field['error'].append(errors)
            field['std'].append(std)
            field['mean'].append(mean)

    return field


def _get_samples(cond, cinn, device, n_samples=100):
    data_size = 2 * math.prod(cond.shape[-2:])
    batch_size = cond.shape[0]
    source = cond[:, :1, ...].to(device)
    # target = cond[:, 1:, ...].to(device)

    fields = []
    predictions = []
    for k in range(batch_size):
        sample= torch.stack([cond[k] for _ in range(n_samples)])

        z = torch.randn(n_samples, data_size).to(device)
        v_field_pred, pred = cinn(z, c=sample, rev=True)
        fields.append(v_field_pred)
        predictions.append(pred)

        # sample_source = torch.stack([source[k] for _ in range(n_samples)])
        # target_pred, _ = apply_flow(v_field=v_field_pred, volume=sample_source, device=device)
        # predictions.append(target_pred)

    return fields, predictions


def _visualize_embedding(fields, save_path=None, method='umap'):
    """
    fields: a list of 2-d vector fields of a
    """
    # pairwise_distances = [mse(fields[i], fields[j]).cpu().detach().numpy().item() for (i, j) in
    # itertools.product(range(len(fields)), repeat=2)]
    # sample_size
    # fields_np = [f.cpu().detach().numpy().reshape((sample_size, -1)) for f in fields]

    sample_size = fields.shape[0]
    fields = fields.cpu().detach().numpy().reshape((sample_size, -1))

    if method == 'pca':
        pca = PCA(2)
        embedding = pca.fit_transform(fields) # call it embedding even though its technically compnents
        print("----- explained variance: ----------- \n", pca.explained_variance_ratio_)
        # components = components[]
        df = pd.DataFrame(embedding)
        sns.displot(df, x=0, y=1, kind='kde')

    elif method == 'tsne':
        tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
        embedding = tsne.fit_transform(fields)
        df = pd.DataFrame(embedding)
        sns.displot(df, x=0, y=1, kind='kde')

    elif method == 'mds':
        mds = MDS(n_components=2)
        embedding = mds.fit_transform(fields)
        df = pd.DataFrame(embedding)
        sns.displot(df, x=0, y=1, kind='kde')

    elif method == 'umap':
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(fields)
        # from sklearn.neighbors import KernelDensity
        # from scipy.stats import gaussian_kde
        # density = gaussian_kde(embedding)

        df = pd.DataFrame(embedding)
        sns.displot(df, x=0, y=1, kind='kde')

    else:
        raise AttributeError("No valid embedding!")

    # fig = plt.scatter(components[:, 0], components[:, 1])
    # sns.displot(components, x=, y=, kind="kde")
    # sns.histplot(pairwise_distances)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return embedding

# def _make_glyph(fields):
#     for f in fields:
#         x_len = f.shape[0]
#         y_len = f.shape[1]
#
#         for x in range(x_len):
#             for y in range(y_len):



def _error_dist(z, cond):
    latent_coord = np.linalg.norm(z, )


