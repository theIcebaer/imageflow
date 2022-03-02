import os
import torch
import numpy as np
import subprocess
import imageflow as imf
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch.layers import VecInt, SpatialTransformer



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

def convert_hdf5():
    import h5py
    import torch

    dat_path = "data/mnist_rnd_distortions_10.dat"
    hdf5_path = "data/mnist_rnd_distortions_10.hdf5"

    dat = torch.load(dat_path)

    with h5py.File(hdf5_path, 'w') as f:
        f['v_field'] = dat['vfield'].type(torch.float)
        f['source'] = dat['original'].type(torch.float)
        f['target'] = dat['transformed'].type(torch.float)

    return True

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    # buf.close()
    plt.close()
    return img

def streamplot_from_batch(batch,epoch, show_image=True, get_all=False, idxs=None, save_path=None, quiver=False):
    import numpy as np
    import matplotlib.pyplot as plt

    if get_all:
        idxs = np.arange(batch.shape[0])
    elif idxs is not None:
        pass
    else:
        idxs = [np.random.randint(0, batch.shape[0]-1)]
    data_shape = batch.shape[2], batch.shape[3]

    images = []
    for idx in idxs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        u, v = batch[idx, 0, :, :], batch[idx, 1, :, :]
        x, y = np.meshgrid(np.linspace(0, data_shape[0]-1, data_shape[1]), np.linspace(0, data_shape[0]-1, data_shape[1]))
        ax.set_aspect('equal', 'box')
        plt.streamplot(x, y, -v, u)

        if len(idxs) > 10:
            print(idx)
        # ax.set_ylim((0, x.shape[1]))
        # ax.set_xlim((0, x.shape[0]))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        # plt.show()
        image = fig2img(fig)
        if save_path:
            path = os.path.join(save_path, f"v_field_{epoch}_{idx}.png")
            image.save(path)

        if show_image:
            image.show()
        images.append(image)

    return images

def quiverplot_from_batch(batch, show_image=True, get_all=False, idxs=None, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if get_all:
        idxs = np.arange(batch.shape[0])
    elif idxs:
        pass
    else:
        idxs = [np.random.randint(0, batch.shape[0]-1)]
    data_shape = batch.shape[2], batch.shape[3]

    images = []
    for idx in idxs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        u, v = batch[idx, 0, :, :], batch[idx, 1, :, :]
        x, y = np.meshgrid(np.linspace(0, data_shape[0]-1, data_shape[1]), np.linspace(0, data_shape[0]-1, data_shape[1]))
        ax.set_aspect('equal', 'box')
        plt.quiver(x, y, u, v)
        if len(idxs) > 10:
            print(idx)
        # ax.set_ylim((0, x.shape[1]))
        # ax.set_xlim((0, x.shape[0]))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        # plt.show()
        image = fig2img(fig)
        if save_path:
            path = os.path.join(save_path, f"v_field_{idx}.png")
            image.save(path)

        if show_image:
            image.show()
        images.append(image)

    return images

def plot_mf(batch):
    import numpy as np
    import matplotlib.pyplot as plt
    idx = np.random.randint(0, batch.shape[0]-1)
    m = batch[idx, 0, ...]
    f = batch[idx, 1, ...]

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(m)
    fig.add_subplot(1, 2, 2)
    plt.imshow(f)
    plt.show()

def make_figures_mf(batch, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    idxs = np.arange(batch.shape[0])

    movings = []
    fixeds = []
    for idx in idxs:
        fig_m = plt.figure(1)
        ax_m = fig_m.add_subplot(111)
        ax_m.get_xaxis().set_visible(False)
        ax_m.get_yaxis().set_visible(False)
        plt.imshow(batch[idx, 0, ...])
        # plt.title(f"moving {idx}")
        im_1 = fig2img(fig_m)
        movings.append(im_1)
        fig_f = plt.figure(2)
        ax_f = fig_f.add_subplot(111)
        ax_f.get_xaxis().set_visible(False)
        ax_f.get_yaxis().set_visible(False)
        plt.imshow(batch[idx, 1, ...])
        # plt.title(f"fixed {idx}")
        im_2 = fig2img(fig_f)
        fixeds.append(im_2)
        # plt.show()
        if save_path:
            im_1.save(os.path.join(save_path, f"moving_{idx}.png"))
            im_2.save(os.path.join(save_path, f"fixed_{idx}.png"))
    return movings, fixeds


def load_birl(path, scale=5, mode='train'):
    """ Small script to load the birl dataset.
    path: path to the birl directory
    scale: data resolution to load
    """
    import glob
    from PIL import Image
    import csv
    test_sets = ['lung-lesion_3', 'lung-lobes_4', 'mammary-gland_2']

    sets = os.listdir(path)
    train_sets = [s for s in sets if s not in test_sets]
    if mode == 'train' or mode == 'validation':
        sets = train_sets
    else:
        sets = test_sets
    images = []
    csvs = []
    # print("sets", sets)
    for i, s in enumerate(sets):
        set_path = os.path.join(path, s, f"scale-{scale}pc")
        image_paths = glob.glob(set_path+"/*.jpg")
        csv_paths = glob.glob(set_path+"/*.csv")

        images.append([])
        csvs.append([])
        for csv_path, im_path in zip(csv_paths, image_paths):

            images[i].append(Image.open(im_path))
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                csv_list = list(reader)
                csvs[i].append(csv_list)

    return images, csvs

def make_split(in_channels, split):
    total_split = sum(split)
    if in_channels > total_split:
        factor = in_channels / total_split
        if not factor == int(factor):
            raise("split is not scalable.")
        split = tuple([int(s * factor) for s in split])

    elif in_channels < total_split:
        factor = total_split / in_channels
        if not factor == int(factor):
            raise ("split is not scalable.")
        split = tuple([int(s/factor) for s in split])
    return split


def make_checkpoint(model, model_type, optimizer, scheduler,
                    data_res):
    scheduler_name = {"<class 'torch.optim.lr_scheduler.MultiStepLR'>": 'MultiStep',
                      "<class 'torch.optim.lr_scheduler.CosineAnnealingLR'>": "Annealing",
                      "<class 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'>": "AnnealingWarmRestarts"
    }

    checkpoint = {}
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['model_type'] = model_type
    checkpoint['model_architecture'] = model.model_architecture
    checkpoint['optimizer_state'] = optimizer.state_dict()
    checkpoint['scheduler_type'] = scheduler_name[str(type(scheduler))]
    checkpoint['scheduler_state'] = scheduler.state_dict()
    checkpoint['data_res'] = data_res
    if model_type == 'basic':
        cond_net = model.cond_net
        checkpoint['cond_net'] = {
            'net_type': cond_net.net_type,
            'pretraining': cond_net.pretraining,
            'out_shape': cond_net.out_shape
        }

    return checkpoint


def make_cinn(model_type, model_architecture=None, init_method='xavier', data_res=(28, 28), **kwargs):
    from imageflow.nets import CinnBasic, CinnConvMultiRes
    import warnings
    print("Configuring model architecture.")

    if model_type == "basic":
        if not model_architecture:
            args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in kwargs.keys()]
            model_architecture = {key: kwargs[key] for key in args}
            if not model_architecture:
                raise AttributeError(f"No model architecture specified. Please specify valid architecture for cinn "
                                     f"{model_type}.")

        if kwargs.get("cond_net"):
            # TODO: conditioning net kann nur als explizites modell (ich glaube als wrapper klasse) übergeben werden.
            #       Wäre schön wenn das auch nur als config string ginge.
            print(f"Using specified conditioning net {type(kwargs.get('cond_net'))}.")
            cinn = CinnBasic(init_method=init_method,
                             cond_net=kwargs.get("cond_net"),
                             **model_architecture)
        else:
            warnings.warn("No conditioning Network specified. Falling back to default.")
            cinn = CinnBasic(init_method=init_method, **model_architecture)

    elif model_type == "multiRes":
        if not model_architecture:
            args = [arg for arg in ["block_depth", "cond_channels", "splits", "downsample", "conv_channels"] if
                    arg in kwargs.keys()]
            model_architecture = {key: kwargs[key] for key in args}
            if not model_architecture:
                raise AttributeError(f"No model architecture specified. Please specify valid architecture for cinn "
                                     f"{model_type}.")
        cinn = CinnConvMultiRes(init_method=init_method, data_res=data_res, **model_architecture)
    else:
        raise AttributeError("No valid model type specified. Use model_type='basic' or model_type='multiRes'.")
        # print("Unknown model type. Falling back to default CinnBasic.")
        # cinn = CinnBasic(init_method=init_method, **model_architecture)

    return cinn


def make_optimizer(cinn, **optimizer_conf):
    train_params = [p for p in cinn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(train_params, **optimizer_conf)
    return optimizer


def make_scheduler(scheduler_type, optimizer, n_epochs=60, **scheduler_conf):
    if scheduler_type == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_conf)
    elif scheduler_type == "Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_conf)
    elif scheduler_type == "AnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_conf)
    else:
        print("No scheduler configured. Falling back to MultiStepLR.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[n_epochs // 3, n_epochs // 1.5])
    return scheduler


def load_checkpoint(checkpoint):
    if checkpoint['model_type'] == 'basic':
        from imageflow.nets import CondNetWrapper
        cond_net_conf = checkpoint['model_architecture'].get('cond_net')
        cond_net = CondNetWrapper(cond_net_conf['net_name'], cond_net_conf['pretraining'])
        model = make_cinn(checkpoint['model_type'], checkpoint['model_architecture'], checkpoint['data_res'],
                          cond_net=cond_net)
    else:
        model = make_cinn(checkpoint['model_type'], checkpoint['model_architecture'], checkpoint['data_res'])
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tmp_var' not in k}
    model.load_state_dict(state_dict)

    optimizer = make_optimizer(model)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    scheduler = make_scheduler(checkpoint['scheduler_type'], checkpoint['scheduler_conf'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    return model, optimizer, scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def make_dataset(path, mode, type, noise=0.08):
    from imageflow.dataset import MnistDataset
    if type == 'MNIST' or 'mnist':
        if mode=='train':
            data_set = MnistDataset(path, noise=noise)  # also try 0.005
            n_samples = len(data_set)
            rng = np.random.default_rng(42)
            indices = rng.permutation(np.arange(n_samples))
            train_indices = indices[:40000]
            val_indices = indices[40000:44000]

            train_set = torch.utils.data.Subset(data_set,
                                                      train_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
            val_set = torch.utils.data.Subset(data_set, val_indices)

            return train_set, val_set
        else:
            raise NotImplemented("Only train mode is supported for function 'make_dataset'.")
    else:
        raise NotImplemented("Only MNIST datase is supported for 'make_dataset'.")

def get_plot_data(path):
    with open(path) as f:
        loss_log = f.readlines()



    nll = [float(l.split()[2]) for l in  loss_log]
    val_nll = [float(l.split()[3]) for l in loss_log]

    reg_loss = [float(l.split()[-1]) for l in loss_log]

    run_avg_nll = [np.mean(nll[i:i+10]) for i, _ in enumerate(loss_log[:-10])]
    run_avg_nll = [*[np.mean(nll[0:i]) for i in range(1,11)], *run_avg_nll]

    run_avg_val = [np.mean(val_nll[i:i+10]) for i, _ in enumerate(loss_log[:-10])]
    run_avg_val = [*[np.mean(val_nll[0:i]) for i in range(1,11)], *run_avg_val]
    run_avg_val = [run_avg_val[i] + reg_loss[i] for i in range(len(reg_loss))]
# print(loss_log[0].split())

    n_datapoints = len(loss_log)
    ticks = [int(n_datapoints / 4), int(n_datapoints / 2), int(n_datapoints * 3 / 4), n_datapoints]
    print(ticks)
    ticks = [int(k) for k in range(1,n_datapoints+1) if k %80 == 0]
    print(ticks)
    epochs = [float(l.split()[0]) for l in  loss_log]
    labels = [int(epochs[x - 1] + 1) for x in ticks]

    print(len(run_avg_val))
    return run_avg_nll, run_avg_val, reg_loss, ticks, labels