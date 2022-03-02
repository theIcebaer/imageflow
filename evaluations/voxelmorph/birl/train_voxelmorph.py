import os
import datetime
import torch
import yaml
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import imageflow.utils

from torch.utils.data import DataLoader
from imageflow.nets import CinnBasic
from imageflow.nets import Reg_mnist_cINN
from imageflow.dataset import MnistDataset



# -- settings ---------------------------------------------------------------------------------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(f"Script running with device {device}.")
batch_size = 128
val_batch_size = 128
test_batch_size = 256
n_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-5
# scheduler_config = {  # multistep scheduler config
#     "scheduler_gamma": 0.1,
#     "scheduler_milestones": [20, 40]
# }
scheduler_config = {  # cosine annealing scheduler config
    "T_max": 5,
}



image_shape = (28, 28)
field_shape = (2, 28, 28)
ndim_total = 28*28*2
plot = True
base_dir = "/home/jens/thesis/imageflow"
run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
data_dir = os.path.join(base_dir, "data")

# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from imageflow.dataset import BirlData
print("Preparing the data loaders...")
data_scale = 5
sample_res = (128, 128)
train_set = BirlData(os.path.join(data_dir, 'birl'), mode='train', color='grayscale', sample_res=sample_res, scale=data_scale)
val_set = BirlData(os.path.join(data_dir, 'birl'), mode='validation', color='grayscale', sample_res=sample_res, scale=data_scale)

train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_set, batch_size=val_batch_size, drop_last=True)
print("...done.")

print("initializing voxelmorph model...")
inshape = (128, 128)#next(iter(train_loader))[1].shape[2:]
nb_unet_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
bidir = False
int_steps = 7
int_downsize = 2

vxm_model = vxm.networks.VxmDense(inshape=inshape,
                                  nb_unet_features=nb_unet_features,
                                  bidir=bidir,
                                  int_steps=int_steps,
                                  int_downsize=int_downsize)
vxm_model.train()
vxm_model.to(device)

config = {}
print("Configuring optimizer and sheduler...")
train_params = [p for p in vxm_model.parameters() if p.requires_grad]
print(sum(p.numel() for p in train_params))
optimizer = torch.optim.Adam(train_params, lr=learning_rate, weight_decay=weight_decay)
# scheduler = None
if config.get("scheduler") == "MultiStep":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get("scheduler_params"))
elif config.get("scheduler") == "Annealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.get("scheduler_params"))
elif config.get("scheduler") == "AnnealingWarmRestarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.get("scheduler_params"))
else:
    print("No scheduler configured. Falling back to MultiStepLR.")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[n_epochs // 3, n_epochs // 1.5])
    config['scheduler'] = "MultiStep"
print("...done.")


print("...done.")





from imageflow.losses import Grad
loss_log = ""
for e in range(n_epochs):
    for i, batch in enumerate(train_loader):
        if i > 300:
            break

        cond = batch
        source = cond[:, :1, ...].to(device)
        target = cond[:, 1:, ...].to(device)
        # print(source.shape)torch.nn.functional.pad(
        # print(target.shape)torch.nn.functional.pad(



        target_pred, v_field_pred = vxm_model(source, target)

        loss = torch.mean(torch.mean(torch.square(target_pred - target), dim=(1, 2, 3)))
        smoother = Grad(penalty='l2', mult=1)
        smooth_reg = smoother.loss(v_field_pred)
        loss += smooth_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 10.)
        optimizer.step()
        line = f"{e} {i} {loss}"
        print(line)
        loss_log+=f"{line}\n"


with open("loss_log.txt", "w") as f:
    f.write(loss_log)



# vxm_model.save("voxelmorph_model.pt")


