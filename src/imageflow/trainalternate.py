import os
import datetime
import sys
import time

import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import imageflow
from imageflow.nets import CinnBasic
from imageflow.nets import CinnConvMultiRes
from imageflow.nets import Reg_mnist_cINN
from imageflow.dataset import MnistDataset
from imageflow.dataset import BirlData
import seaborn as sns
import matplotlib.pyplot as plt

from imageflow.visualize import streamplot_from_batch, plot_mff_
import os
import datetime
import torch
import yaml

from torch.utils.data import DataLoader
import imageflow.utils
# from imageflow.nets import CinnBasic
# from imageflow.nets import Reg_mnist_cINN
from imageflow.nets import CinnConvMultiRes
from imageflow.dataset import MnistDataset


def train_unsupervised(n_epochs=10, batch_size=256, val_batch_size=16, device=None, sup_data="MNIST_small", unsup_data='MNIST', learning_rate=5e-4,
                       weight_decay=1e-5,
                       sup_file_name="mnist_rnd_distortions_10.hdf5",
                       unsup_file_name="mnist_rnd_distortions_1.hdf5",**config):
    """
    Wie das ursprüngliche unsupervised learning nur mit multi resolution setup.
    """

    # -- settings ------------------------------------------------------------------------------------------------------
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print(f"Script running with device {device}.")
    # batch_size = 256
    # val_batch_size = 256
    # test_batch_size = 256
    # n_epochs = 10
    # learning_rate = 1e-4
    # weight_decay = 1e-5
    scheduler_config = {  # multistep scheduler config
        "scheduler_gamma": 0.1,
        "scheduler_milestones": [20, 40]
    }
    # scheduler_config = {  # cosine annealing scheduler config
    #     "T_max": 5,
    # }

    # init_method = 'gaussian'

    augm_sigma = 0.08

    # image_shape = (28, 28)
    # field_shape = (2, 28, 28)

    # plot = True
    if config.get("base_dir"):
        base_dir = config.get("base_dir")
    else:
        base_dir = "/home/jens/thesis/imageflow"
    if config.get("run_dir"):
        run_dir = config.get("run_dir")
    else:
        run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if config.get('sup_data_dir'):
        sup_data_dir = config.get("data_dir")
    else:
        sup_data_dir = os.path.join(base_dir, "data")
    if config.get('unsup_data_dir'):
        unsup_data_dir = os.path.join(base_dir, "data")
    else:
        unsup_data_dir = os.path.join(base_dir, "data")
    # ---------------------------------------------------------------------------------------------------------------------

    print("Preparing Datasets...")
    if config.get('data_res'):
        data_res = config.get('data_res')
    else:
        data_res = (28, 28)
    ndim_total = int(np.prod(data_res) * 2)
    print(f"Data resolution is {data_res}.")
    # if data == "Birl":
    #     print("Initializing Birl dataset")
    #     data_scale = config.get("birl_scale")
    #     train_set = BirlData(os.path.join(data_dir, 'birl'), mode='train', color='grayscale', sample_res=data_res, scale=data_scale)
    #     val_set = BirlData(os.path.join(data_dir, 'birl'), mode='validation', color='grayscale', sample_res=data_res, scale=data_scale)
    if unsup_data=="MNIST":  # if data == "MNIST":
        print("Initializing MNIST data.")
        from imageflow.utils import make_dataset
        unsup_train_set, unsup_val_set = make_dataset(os.path.join(unsup_data_dir, unsup_file_name),
                                                      mode='train', type='MNIST')
    else:
        raise AttributeError("No valid dataset for unsupervised training.")
    shuffle = True
    unsup_train_loader = DataLoader(unsup_train_set, batch_size=batch_size, drop_last=True, shuffle=shuffle)
    unsup_val_loader = DataLoader(unsup_val_set, batch_size=val_batch_size, drop_last=True, shuffle=shuffle)
    if sup_data == "MNIST_small":
        sup_train_set, sup_val_set = make_dataset(os.path.join(sup_data_dir, sup_file_name),
                                                  mode='train', type='MNIST')
    else:
        raise AttributeError("No valid dataset specified.")

    sup_train_loader = DataLoader(sup_train_set, batch_size=batch_size, drop_last=True, shuffle=shuffle)
    sup_val_loader = DataLoader(sup_val_set, batch_size=val_batch_size, drop_last=True, shuffle=shuffle)
    print("Initializing Data loaders.")




    print("...done.")

    # print("initializing cINN...")
    # cinn = CinnConvMultiRes()
    # cinn.to(device)
    # cinn.train()
    # print("...done.")

    print("Initializing cINN...")
    if config.get("pretrained_model"):
        print("Loaded model is given. Skip model construction and use given model.")
        cinn = config.get("pretrained_model")
        raise AttributeError("Pretrained model should be given as path/to/checkpoint")
    else:
        print("Configuring model architecture.")
        args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in config.keys()]
        model_architecture = {key: config[key] for key in args}

        init_method = "xavier"
        if config.get("init_method"):
            init_method = config.get("init_method")

        if config.get("model_checkpoint"):
            checkpoint = config.get("model_checkpoint")
            model_type = checkpoint["model_type"]

        elif config.get("model_type"):
            model_type = config.get("model_type")

        else:
            model_type = "basic"

        if model_type == "basic":
            if config.get("cond_net"):
                cinn = CinnBasic(init_method=init_method,
                                 cond_net=config.get("cond_net"),
                                 **model_architecture)
            else:
                cinn = CinnBasic(init_method=init_method, **model_architecture)
        elif model_type == "multiRes":
            args = [arg for arg in ["block_depth", "cond_channels", "splits", "downsample", "conv_channels"] if
                    arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            cinn = CinnConvMultiRes(init_method=init_method, data_res=data_res, **model_architecture)
        else:
            print("Unknown model type. Falling back to default CinnBasic.")
            model_type = 'basic'
            cinn = CinnBasic(init_method=init_method, **model_architecture)

    # cinn = Reg_mnist_cINN(device=device)
    cinn.to(device)
    cinn.train()
    print("...done.")

    print("Configuring optimizer and sheduler...")
    train_params = [p for p in cinn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(train_params, lr=learning_rate, weight_decay=weight_decay)
    # scheduler = None
    if config.get("scheduler") == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.get("scheduler_params"))
    elif config.get("scheduler") == "Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config.get("scheduler_params"))
    elif config.get("scheduler") == "AnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config.get("scheduler_params"))
    else:
        print("No scheduler configured. Falling back to MultiStepLR.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[n_epochs // 3, n_epochs // 1.5])
        config['scheduler'] = "MultiStep"
    print("...done.")

    if config.get("checkpoint"):
        # If we have a pretrained model given as path to model state dict.
        print(f"Loading checkpoint data from file...{config.get('checkpoint')}")
        model_dir = config.get("checkpoint")
        model_dict = torch.load(model_dir)
        state_dict = {k: v for k, v in model_dict['state_dict'].items() if 'tmp_var' not in k}
        cinn.load_state_dict(state_dict)
        optimizer.load_state_dict(model_dict["optimizer_state"])
        if config.get("continue_run"):
            scheduler.load_state_dict(model_dict["scheduler_state"])


    print("preparing run directory...")
    os.mkdir(run_dir)
    os.mkdir(os.path.join(run_dir, "checkpoints/"))
    print("...done")

    output_log = ""
    loss_log = {}
    loss_log['nll'] = []
    loss_log['val_nll'] = []
    loss_log['epoch'] = []
    loss_log["batch"] = []
    loss_log['lr'] = []

    n_batches_per_epoch = batch_size/len(unsup_train_set)

    print("epoch \t batch \t train_loss \t val_loss")#\t rec_loss \t prior_loss \t jac_loss")
    for e in range(n_epochs):
        agg_nll = []
        # if e > 0: break
        sup_iterator = iter(sup_train_loader)
        unsup_iterator = iter(unsup_train_loader)
        for i in n_batches_per_epoch:
            #unsupervised training step
            from imageflow.training_steps import unsupervised_training_step
            unsupervised_training_step(e, i, unsup_iterator, unsup_val_loader, cinn, train_params, optimizer, scheduler, ndim_total, device, output_log, **config)

            # supervised training step
            from imageflow.training_steps import supervised_train_step
            supervised_train_step(e,i, sup_iterator, sup_val_loader, cinn, train_params, optimizer, scheduler, ndim_total, device, loss_log, **config)

            

        if e % 10 == 0:
            from imageflow.utils import make_checkpoint
            checkpoint = make_checkpoint(cinn, model_type, optimizer, scheduler, data_res)
            torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}.pt'.format(e)))

        scheduler.step()

    checkpoint = {
        "state_dict": cinn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_final.pt'))

    with open(os.path.join(run_dir, 'loss.log'), 'w') as log_file:
        log_file.write(output_log)
    with open(os.path.join(run_dir, 'params.yaml'), 'w') as params_file:
        commit, link = imageflow.utils.get_commit()
        device_str = str(device)
        params_yml = {
            "device": device_str,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_config": scheduler_config,
            "commit": commit,
            "git-repo": link
        }
        doc = yaml.dump(params_yml, params_file)
