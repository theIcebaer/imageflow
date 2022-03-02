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
from imageflow.dataset import FireDataset

def train_supervised(n_epochs=10, batch_size=512, val_batch_size=1024, device=None, data='MNIST', init_params=None,
                     cond_net=None, file_name="mnist_rnd_distortions_1.hdf5", **config):
    if not device:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Script running with device {device}.")
    # milestones = [20, 40]
    # image_shape = (28, 28)
    # field_shape = (2, 28, 28)
    ndim_total = 28*28*2
    # plot = True
    base_dir = "../../"
    run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    data_dir = os.path.join(base_dir, "data")
    if config.get("run_dir"):
        run_dir = config.get("run_dir")
    if config.get("data_dir"):
        data_dir = config.get("data_dir")

    print("Preparing Datasets...")
    if config.get('data_res'):
        data_res = config.get('data_res')
    else:
        data_res = (28, 28)
    ndim_total = int(np.prod(data_res) * 2)
    print(f"Data resolution is {data_res}.")

    if data == 'MNIST':
        print("Initializing MNIST data.")
        data_set = MnistDataset(os.path.join(data_dir, file_name), noise=0.08)  # also try 0.005
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        train_indices = indices[:40000]
        val_indices = indices[40000:44000]
        if config.get("semi_supervised"):
            train_indices = train_indices[:10000]

        train_set = torch.utils.data.Subset(data_set, train_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
        val_set = torch.utils.data.Subset(data_set, val_indices)
    else:
        print("Initializing Birl data.")
        if config.get('birl_scale'):
            birl_scale= config.get('birl_scale')
        train_set = BirlData(os.path.join(data_dir, 'birl'), mode='train', color='grayscale', sample_res=data_res, scale=birl_scale)
        val_set = BirlData(os.path.join(data_dir, 'birl'), mode='validation', color='grayscale', sample_res=data_res, scale=birl_scale)

    print("Initializing data loaders.")
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print("...done.")

    print("Initializing cINN...")
    init_method = "xavier"
    if config.get("init_method"):
        init_method = config.get("init_method")
        print(f"Weights are initialized with {init_method} method")
    if init_params:
        print(f"method parameters: {init_params}")

    if config.get("model_type"):
        model_type = config.get("model_type")
        if config.get("model_type") == "basic":
            args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            # print(f"Creating basic cinn with conditioning network {type(cond_net)}")
            # Model = getattr(sys.modules[imageflow.nets], )
            cinn = CinnBasic(init_method=init_method,
                             init_params=init_params,
                             cond_net=cond_net,
                             **model_architecture)

        elif config.get("model_type") == "multiRes":
            args = [arg for arg in ["block_depth", "cond_channels", "splits", "downsample", "conv_channels"] if arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            cinn = CinnConvMultiRes(init_method=init_method, **model_architecture)
        else:
            print("Unknown model type. Falling back to default CinnBasic.")
            args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in config.keys()]
            model_architecture = {key: config[key] for key in args}
            cinn = CinnBasic(init_method=init_method, init_params=init_params, **model_architecture)
    else:
        print("No model type configured. Default is CinnBasic.")
        model_type = 'basic'
        args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in config.keys()]
        model_architecture = {key: config[key] for key in args}
        cinn = CinnBasic(init_method=init_method, init_params=init_params, **model_architecture)

    # cinn = Reg_mnist_cINN(device=device)
    cinn.to(device)
    cinn.train()
    print("...done.")

    print("Configuring optimizer and scheduler...")
    train_params = [p for p in cinn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(train_params, lr=5e-4, weight_decay=1e-5)
    if config.get("optimizer_checkpoint"):
        checkpoint = config.get("optimizer_checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
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
    if config.get("scheduler_checkpoint"):
        checkpoint = config.get("scheduler_checkpoint")
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    print("...done.")


    print("Preparing logging...")
    loss_log = {"nll": [],
                "val_nll": [],
                "epoch": [],
                "batch": [],
                "lr": []}
    print("...done.")

    print("preparing run directory...")
    os.mkdir(run_dir)
    os.mkdir(os.path.join(run_dir, "checkpoints/"))
    print("...done.")

    if not config.get("suppress_loss_print"):
        print("epoch \t batch \t nll \t aggregated nll \t validation nll")

    for e in range(n_epochs):
        agg_nll = []
        if config.get("testrun"):
            if e > 0:
                break
        for i, (im, cond) in enumerate(train_loader):
            if config.get("testrun"):
                if i > 0:
                    break
            im = im.to(device)
            cond = cond.to(device)

            out, log_j = cinn(im, c=cond)

            alt_nll = torch.mean(out ** 2 / 2) - torch.mean(log_j) / ndim_total

            nll = 0.5 * torch.sum(out ** 2, dim=1) - log_j
            nll = torch.mean(nll) / ndim_total

            if config.get("grad_loss"):
                from imageflow.losses import Grad
                penalty = config.get('grad_loss')['penalty']
                grad_lambda = config.get("grad_loss")["multiplier"]
                smoother = Grad(penalty=penalty, mult=grad_lambda)
                smooth_reg = smoother.loss(out)
                nll += smooth_reg

            alt_nll.backward()

            # print("{}\t{}\t{}".format(e, i, alt_nll.item()))

            torch.nn.utils.clip_grad_norm_(train_params, 100.)

            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

            if torch.isnan(alt_nll):
                break

            if i % 20 == 0:
                with torch.no_grad():
                    val_x, val_c = next(iter(val_loader))
                    val_x, val_c = val_x.to(device), val_c.to(device)

                    v_out, v_log_j = cinn(val_x, c=val_c)
                    v_nll = torch.mean(v_out ** 2) / 2 - torch.mean(v_log_j) / ndim_total
                    loss_log['nll'].append(alt_nll.item())
                    loss_log['val_nll'].append(v_nll.item())
                    loss_log['epoch'].append(e)
                    loss_log["batch"].append(i)
                    loss_log['lr'].append(scheduler.get_last_lr())
                    print("{}\t{}\t{}\t{}\t{}\t{}".format(e, i, alt_nll.item(), nll.item(), v_nll.item(),
                                                          scheduler.get_last_lr()))
                    agg_nll.append(alt_nll)

            if i == 0 and e % 10 == 0:
                if not config.get("suppress_output_online"):

                    checkpoint = {
                        "model_type": type(cinn).__name__,
                        "state_dict": cinn.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    }

                    torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}_{}.pt'.format(e, i)))
                    torch.save(loss_log, os.path.join(run_dir, 'checkpoints/loss_log.pt'))
        scheduler.step()
        if torch.isnan(agg_nll[-1]):
            break

    checkpoint = {
        "model_type": type(cinn).__name__,
        "state_dict": cinn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_final.pt'))
    torch.save(loss_log, os.path.join(run_dir, 'checkpoints/loss_log.pt'))

    return cinn, loss_log["nll"][-1], loss_log["val_nll"][-1]

# config = {
#     "subnet_depth": 2,
#     "subnet_width": 1024,
#     "model_depth": 20
# }
#
# train_supervised(subnet_depth=2, subnet_width=1024, model_depth=20)


def train_unsupervised(n_epochs=10, batch_size=256, val_batch_size=8, device=None, data='MNIST', learning_rate=5e-4,
                       weight_decay=1e-5, file_name="mnist_rnd_distortions_1.hdf5", **config):
    """
    Wie das ursprüngliche unsupervised learning nur mit multi resolution setup.
    """
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
        base_dir = "../../"
    if config.get("run_dir"):
        run_dir = config.get("run_dir")
    else:
        run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if config.get('data_dir'):
        data_dir = config.get("data_dir")
    else:
        data_dir = os.path.join(base_dir, "data")

    # ---------------------------------------------------------------------------------------------------------------------

    print("Preparing Datasets...")
    if config.get('data_res'):
        data_res = config.get('data_res')
    else:
        data_res = (28, 28)
    ndim_total = int(np.prod(data_res) * 2)
    print(f"Data resolution is {data_res}.")
    if data == "Birl":
        print("Initializing Birl dataset")
        data_scale = config.get("birl_scale")
        sample_res = config.get('image_res') if config.get("image_res") else data_res
        train_set = BirlData(os.path.join(data_dir, 'birl'), mode='train', color='grayscale', sample_res=sample_res, scale=data_scale)
        val_set = BirlData(os.path.join(data_dir, 'birl'), mode='validation', color='grayscale', sample_res=sample_res, scale=data_scale)
    elif data == "MNIST":
        print("Initializing MNIST data.")
        data_set = MnistDataset(os.path.join(data_dir, file_name), noise=0.08)  # also try 0.005
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        train_indices = indices[:40000]
        val_indices = indices[40000:44000]
        if config.get("semi_supervised"):
            train_indices = train_indices[10000:]
        train_set = torch.utils.data.Subset(data_set,
                                            train_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
        val_set = torch.utils.data.Subset(data_set, val_indices)

    elif data=="FIRE":
        size = config.get("image_res")[0] if config.get("image_res") else 184
        data_set = FireDataset(data_dir, size=size)
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        train_indices = indices[:100]
        val_indices = indices[100:108]

        train_set = torch.utils.data.Subset(data_set,
                                            train_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
        val_set = torch.utils.data.Subset(data_set, val_indices)

    elif data=="Covid":
        from imageflow.dataset import CovidXDataset
        data_set = CovidXDataset(data_dir)
        n_samples = len(data_set)
        rng = np.random.default_rng(42)
        indices = rng.permutation(np.arange(n_samples))
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.1)
        train_indices = indices[:n_train]
        val_indices = indices[n_train: n_train+n_val]

        train_set = torch.utils.data.Subset(data_set,
                                            train_indices)  # random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
        val_set = torch.utils.data.Subset(data_set, val_indices)

    else:
        raise AttributeError("No valid dataset specified.")
    print("Initializing Data loaders.")

    if data == "Birl":
        shuffle = False
    else:
        shuffle = True
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, drop_last=True, shuffle=shuffle)
    val_iter = iter(val_loader)
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
            image_res = config.get('image_res')
            cinn = CinnConvMultiRes(init_method=init_method, data_res=data_res, image_res=image_res, **model_architecture)
        else:
            print("Unknown model type. Falling back to default CinnBasic.")
            model_type = 'basic'
            cinn = CinnBasic(init_method=init_method, **model_architecture)

    print(f"Total model parameter count:{sum(p.numel() for p in cinn.parameters() if p.requires_grad)}")

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

    print("epoch \t batch \t train_loss \t val_loss")#\t rec_loss \t prior_loss \t jac_loss")
    for e in range(n_epochs):
        agg_nll = []
        # if e > 0: break
        for i, batch in enumerate(train_loader):
            if data == "Birl" or data == "FIRE" or data=="Covid":
                cond = batch
            else:
                _, cond = batch
            if data == "Birl":
                if config.get("n_batches"):
                    if i > config.get("n_batches"): break
                else:
                    raise KeyError("Number of batches per epoch not configured. Use e.g. n_batches=300")

            # print(i, "--------------------------------------------" )

            cond = cond.to(device)
            # source = cond[:, :1, ...].to(device)
            target = cond[:, 1:, ...].to(device)

            z = torch.randn(batch_size, ndim_total).to(device)
            target_pred, v_field_pred, log_jac = cinn.reverse_sample(z, cond)


            rec_term = torch.mean(torch.mean(torch.square(target_pred - target), dim=(1, 2, 3)))

            z_prior, prior_jac = cinn(v_field_pred, c=cond)

            prior_nll = torch.mean(torch.sum(z_prior ** 2, dim=1) / 2)
            prior_jac = torch.mean(prior_jac)
            prior_term = (prior_nll / ndim_total - prior_jac)

            jac_term = torch.mean(log_jac)



            loss = rec_term  # + prior_term - jac_term
            if config.get("grad_loss"):
                from imageflow.losses import Grad
                # print("check")
                penalty = config.get('grad_loss')['penalty']
                grad_lambda = config.get("grad_loss")["multiplier"]
                smoother = Grad(penalty=penalty, mult=grad_lambda)
                smooth_reg = smoother.loss(v_field_pred)
                loss += smooth_reg
            if config.get("curl"):
                from imageflow.losses import Rotation
                multiplier = config.get("curl")["multiplier"]
                curl_loss = Rotation(multiplier)
                smooth_reg = curl_loss.loss(v_field_pred)
                loss += smooth_reg

            # rec_out = round(rec_term.item(), 2)
            # prior_out = round(prior_term.item(), 2)
            # p_1_out = round(prior_nll.item(), 2)
            # p_2_out = round(prior_jac.item(), 2)
            # jac_out = round(jac_term.item(), 2)

            # t = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # f = t - a
            #  = {} + {} - {} | {} "
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 10.)
            optimizer.step()

            if i % 20 == 0 or (i % 5 == 0 and data == "FIRE"):

                if data == "Birl" or data == "FIRE" or data=="Covid":
                    val_cond = next(iter(val_loader))  # Is ok like this, since it returns random anyway.
                else:
                    _, val_cond = next(iter(val_loader))
                val_cond = val_cond.to(device)
                val_target = val_cond[:, 1:, ...].to(device)
                z = torch.randn(val_batch_size, ndim_total).to(device)
                val_target_pred, val_v_field_pred, val_log_jac = cinn.reverse_sample(z, val_cond)
                val_rec_term = torch.mean(torch.mean(torch.square(val_target_pred - val_target), dim=(1, 2, 3)))
                val_loss = val_rec_term.item()

                if config.get("grad_loss"):
                    from imageflow.losses import Grad
                    # print("check")
                    penalty = config.get('grad_loss')['penalty']
                    grad_lambda = config.get("grad_loss")["multiplier"]
                    smoother = Grad(penalty=penalty, mult=grad_lambda)
                    smooth_reg = smoother.loss(val_v_field_pred)
                    val_loss += smooth_reg
                if config.get("curl"):
                    from imageflow.losses import Rotation
                    multiplier = config.get("curl")["multiplier"]
                    curl_loss = Rotation(multiplier)
                    smooth_reg = curl_loss.loss(val_v_field_pred)
                    val_loss += smooth_reg


                loss_out = loss.item()
                if loss_out == 0.00 or val_loss == 0.00:
                    loss_out = "{:.1e}".format(loss.item())
                    val_loss = "{:.1e}".format(val_rec_term.item())
                output = "{}\t\t{}\t\t{}\t\t{}".format(e, i, loss_out, val_loss)  # , rec_out, prior_out, jac_out, f)
                if config.get("grad_loss") or config.get('curl'):
                    output += f"\t\t{smooth_reg}"
                output_log += (output + "\n")
                print(output)
                if config.get("show_running"):
                    from imageflow.visualize import plot_running_fields
                    # streamplot_from_batch(val_v_field_pred.cpu().detach().numpy())
                    plot_running_fields(val_v_field_pred.cpu().detach().numpy(), idx=0)
                    plot_mff_(val_cond.cpu().detach().numpy(), val_target_pred.cpu().detach().numpy(), idx=0)

                # streamplot_from_batch(v_field_pred.cpu().detach().numpy())
                # time.sleep(5)
                # plt.close()
                # plot_mff_(cond.cpu().detach().numpy(), target_pred.cpu().detach().numpy())
                # time.sleep(5)
                # plt.close()
                if config.get("dummy_run"):
                    break
        if e % 10 == 0:
            from imageflow.utils import make_checkpoint
            checkpoint = make_checkpoint(cinn, model_type, optimizer, scheduler, data_res)
            torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}.pt'.format(e)))

        scheduler.step()
        if config.get("dummy_run"):
            break

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

    #     # if e > 0: break
    #     for i, batch in enumerate(train_loader):
    #         if data == "Birl" or "FIRE":
    #             cond = batch
    #         else:
    #             _, cond = batch
    #         if data == "Birl":
    #             if config.get("n_batches"):
    #                 if i > config.get("n_batches"): break
    #             else:
    #                 raise KeyError("Number of batches per epoch not configured. Use e.g. n_batches=300")
    #
    #         # print(i, "--------------------------------------------" )
    #
    #         cond = cond.to(device)
    #         # source = cond[:, :1, ...].to(device)
    #         target = cond[:, 1:, ...].to(device)
    #
    #         z = torch.randn(batch_size, ndim_total).to(device)
    #         target_pred, v_field_pred, log_jac = cinn.reverse_sample(z, cond)
    #
    #
    #         rec_term = torch.mean(torch.mean(torch.square(target_pred - target), dim=(1, 2, 3)))
    #
    #         z_prior, prior_jac = cinn(v_field_pred, c=cond)
    #
    #         prior_nll = torch.mean(torch.sum(z_prior ** 2, dim=1) / 2)
    #         prior_jac = torch.mean(prior_jac)
    #         prior_term = (prior_nll / ndim_total - prior_jac)
    #
    #         jac_term = torch.mean(log_jac)
    #
    #
    #
    #         loss = rec_term  # + prior_term - jac_term
    #         if config.get("grad_loss"):
    #             from imageflow.losses import Grad
    #             # print("check")
    #             penalty = config.get('grad_loss')['penalty']
    #             grad_lambda = config.get("grad_loss")["multiplier"]
    #             smoother = Grad(penalty=penalty, mult=grad_lambda)
    #             smooth_reg = smoother.loss(v_field_pred)
    #             loss += smooth_reg
    #         if config.get("curl"):
    #             from imageflow.losses import Rotation
    #             multiplier = config.get("curl")["multiplier"]
    #             curl_loss = Rotation(multiplier)
    #             smooth_reg = curl_loss.loss(v_field_pred)
    #             loss += smooth_reg
    #
    #         # rec_out = round(rec_term.item(), 2)
    #         # prior_out = round(prior_term.item(), 2)
    #         # p_1_out = round(prior_nll.item(), 2)
    #         # p_2_out = round(prior_jac.item(), 2)
    #         # jac_out = round(jac_term.item(), 2)
    #
    #         # t = torch.cuda.get_device_properties(0).total_memory
    #         # r = torch.cuda.memory_reserved(0)
    #         # a = torch.cuda.memory_allocated(0)
    #         # f = t - a
    #         #  = {} + {} - {} | {} "
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(train_params, 10.)
    #         optimizer.step()
    #
    #         if i % 20 == 0:
    #
    #             if data == "Birl":
    #                 val_cond = next(iter(val_loader))  # Is ok like this, since it returns random anyway.
    #             else:
    #                 _, val_cond = next(iter(val_loader))
    #             val_cond = val_cond.to(device)
    #             val_target = val_cond[:, 1:, ...].to(device)
    #             z = torch.randn(val_batch_size, ndim_total).to(device)
    #             val_target_pred, val_v_field_pred, val_log_jac = cinn.reverse_sample(z, val_cond)
    #             val_rec_term = torch.mean(torch.mean(torch.square(val_target_pred - val_target), dim=(1, 2, 3)))
    #             val_loss = round(val_rec_term.item(), 2)
    #             loss_out = round(loss.item(), 2)
    #             if loss_out == 0.00 or val_loss == 0.00:
    #                 loss_out = "{:.1e}".format(loss.item())
    #                 val_loss = "{:.1e}".format(val_rec_term.item())
    #             output = "{}\t\t{}\t\t{}\t\t{}".format(e, i, loss_out, val_loss)  # , rec_out, prior_out, jac_out, f)
    #             if config.get("grad_loss") or config.get('curl'):
    #                 output += f"\t\t{smooth_reg}"
    #             output_log += (output + "\n")
    #             print(output)
    #             if config.get("show_running"):
    #                 from imageflow.visualize import plot_running_fields
    #                 # streamplot_from_batch(val_v_field_pred.cpu().detach().numpy())
    #                 plot_running_fields(val_v_field_pred.cpu().detach().numpy(), idx=0)
    #                 plot_mff_(val_cond.cpu().detach().numpy(), val_target_pred.cpu().detach().numpy(), idx=0)
    #
    #             # streamplot_from_batch(v_field_pred.cpu().detach().numpy())
    #             # time.sleep(5)
    #             # plt.close()
    #             # plot_mff_(cond.cpu().detach().numpy(), target_pred.cpu().detach().numpy())
    #             # time.sleep(5)
    #             # plt.close()
    #     if e % 10 == 0:
    #         from imageflow.utils import make_checkpoint
    #         checkpoint = make_checkpoint(cinn, model_type, optimizer, scheduler, data_res)
    #         torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}.pt'.format(e)))
    #
    #     scheduler.step()
    #
    # checkpoint = {
    #     "state_dict": cinn.state_dict(),
    #     "optimizer_state": optimizer.state_dict(),
    #     "scheduler_state": scheduler.state_dict(),
    # }
    # torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_final.pt'))
    #
    # with open(os.path.join(run_dir, 'loss.log'), 'w') as log_file:
    #     log_file.write(output_log)
    # with open(os.path.join(run_dir, 'params.yaml'), 'w') as params_file:
    #     commit, link = imageflow.utils.get_commit()
    #     device_str = str(device)
    #     params_yml = {
    #         "device": device_str,
    #         "batch_size": batch_size,
    #         "val_batch_size": val_batch_size,
    #         "n_epochs": n_epochs,
    #         "learning_rate": learning_rate,
    #         "weight_decay": weight_decay,
    #         "scheduler_config": scheduler_config,
    #         "commit": commit,
    #         "git-repo": link
    #     }
    #     doc = yaml.dump(params_yml, params_file)
