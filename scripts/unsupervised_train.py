# -*- coding: utf-8 -*-
"""setup_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fabj3TRxAuRU00-ePGmGtgdYAaCvSXmx

Dies hier ist ein kleines experiment, dass überprüfen soll, ob ein cinn auch mit einer Mischung aus GAN setup und
reconstruction loss trainiert werden kann.
Dazu wird zunächst ein sample für ein Geschwindigkeitsfeld generiert angewendet und an ein reconstruction loss
berechnet. Die erste wichtige frage ist, ob anschließend noch ein nll rückwärts berechnet werden muss.
"""

import os
import datetime
import time

import torch
import yaml

from torch.utils.data import DataLoader
import imageflow.utils
from imageflow.nets import CinnBasic
from imageflow.nets import Reg_mnist_cINN
from imageflow.dataset import MnistDataset

# -- settings ---------------------------------------------------------------------------------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(f"Script running with device {device}.")
batch_size = 64
val_batch_size = 1024
test_batch_size = 256
n_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-5
scheduler_config = {  # multistep scheduler config
    "scheduler_gamma": 0.1,
    "scheduler_milestones": [20, 40]
}
# scheduler_config = {  # cosine annealing scheduler config
#     "T_max": 5,
# }

init_method = 'xavier'


augm_sigma = 0.08

image_shape = (28, 28)
field_shape = (2, 28, 28)
ndim_total = 28*28*2
plot = True
base_dir = "../"
run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
data_dir = os.path.join(base_dir, "data")

# ---------------------------------------------------------------------------------------------------------------------

print("Preparing the data loaders...")
data_set = MnistDataset(os.path.join(data_dir, "mnist_rnd_distortions_1.hdf5"))
train_set, val_set, test_set = torch.utils.data.random_split(data_set, [47712, 4096, 8192],
                                                             generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_set, batch_size=val_batch_size, drop_last=True, shuffle=False)
test_loader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True)
print("...done.")

print("initializing cINN...")
cinn = CinnBasic(device=device, init_method=init_method)
cinn.to(device)
cinn.train()
print("...done.")

train_params = [p for p in cinn.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(train_params, lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_config['scheduler_milestones'],
                                                 gamma=scheduler_config['scheduler_gamma'])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_config['T_max'])

print("preparing run directory...")
os.mkdir(run_dir)
os.mkdir(os.path.join(run_dir, "checkpoints/"))
print("...done")

output_log = ""

print("epoch \t batch \t total_loss \t rec_loss \t prior_loss \t jac_loss")
for e in range(n_epochs):
    agg_nll = []
    # if e > 0: break
    for i, (v_field, cond) in enumerate(train_loader):
        # if i : break
        # print(i, "--------------------------------------------" )

        cond = cond.to(device)
        source = cond[:, :1, ...].to(device)
        target = cond[:, 1:, ...].to(device)

        z = torch.randn(batch_size, ndim_total).to(device)
        target_pred, v_field_pred, log_jac = cinn.reverse_sample(z, cond)

        from imageflow.visualize import streamplot_from_batch
        # streamplot_from_batch(v_field_pred.cpu().detach().numpy())
        # plt.close()
        # plot_ff_(target, target_pred)
        # plot_mff_(cond.cpu().detach().numpy(), target_pred.cpu().detach().numpy())
        # plt.close

        rec_term = torch.mean(torch.sum(torch.square(target_pred - target), dim=(1, 2, 3)))

        z_prior, prior_jac = cinn(v_field_pred, c=cond)

        prior_nll = torch.mean(torch.sum(z_prior**2, dim=1) / 2)
        prior_jac = torch.mean(prior_jac)
        prior_term = (prior_nll/ndim_total - prior_jac)

        jac_term = torch.mean(log_jac)

        loss = rec_term # + prior_term - jac_term

        rec_out = round(rec_term.item(), 2)
        prior_out = round(prior_term.item(), 2)
        p_1_out = round(prior_nll.item(), 2)
        p_2_out = round(prior_jac.item(), 2)
        jac_out = round(jac_term.item(), 2)
        loss_out = round(loss.item(), 2)

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = t - a

        output = "{}\t{}\t{} = {} + {} - {} | {} ".format(e, i, loss_out, rec_out, prior_out, jac_out, f)
        output_log += (output + "\n")
        print(output)

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(train_params, 10.)
        loss.backward()
        optimizer.step()

        if i == 0:
            checkpoint = {
                "model_type": type(cinn).__name__,
                "state_dict": cinn.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}_{}.pt'.format(e, i)))
            # from imageflow.utils import streamplot_from_batch, plot_mff_
            import matplotlib.pyplot as plt
            # import numpy as np
            plot_dir = run_dir
            streamplot_from_batch(v_field_pred.cpu().detach().numpy(), save_path=f"running_plots", show_image=True, idxs=0)
            streamplot_from_batch(v_field.cpu().detach().numpy(), save_path=f"running_plots", show_image=True, idxs=0)
            time.sleep(5)
            plt.close()
            imageflow.utils.plot_mff_(cond.cpu().detach().numpy(), target_pred.cpu().detach().numpy(), idx=0)
            time.sleep(5)
            plt.close()

    scheduler.step()

checkpoint = {
    "model_type": type(cinn).__name__,
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
        "model": type(cinn).__name__,
        "device": device_str,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_config": scheduler_config,
        "init_method": init_method,
        "commit": commit,
        "git-repo": link
    }
    doc = yaml.dump(params_yml, params_file)
