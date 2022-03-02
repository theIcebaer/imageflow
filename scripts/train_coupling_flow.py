import os
import datetime
import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from imageflow.nets import CinnBasic
from imageflow.nets import CinnConvMultiRes
from imageflow.nets import Reg_mnist_cINN
from imageflow.nets import CouplingFlow
from imageflow.dataset import MnistDataset
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Script running with device {device}.")
batch_size = 1024
val_batch_size = 1024
n_epochs = 30

augm_sigma = 0.08

image_shape = (28, 28)
ndim_total = 28*28*1
plot = True
base_dir = "../"
run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
data_dir = os.path.join(base_dir, "data")


print("Preparing the data loaders...")
data_set = MnistDataset(os.path.join(data_dir, "mnist_rnd_distortions_1.hdf5"))
train_set, val_set, test_set = random_split(data_set, [47712, 4096, 8192], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=64)
val_loader = DataLoader(val_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)
print("...done.")

print("Initializing cINN...")
# cinn = CinnBasic(init_method="gaussian")
cinn = CouplingFlow(device=device)
cinn.to(device)
cinn.train()
print("...done")

print("trying to plot weights")
train_params = [p for p in cinn.parameters() if p.requires_grad]
# for p in train_params:
#     # torch.randn_like(p).cpu().detach().numpy().flatten() #
#     # dat = torch.randn((1000000))
#     # dat = np.random.randn(1000000, )
#     dat = p.data.cpu()
#     # torch.nn.init.normal_(dat, std=0.1)
#     sns.histplot(dat.detach().numpy().flatten())
#     plt.show()
#     print(dat)
#     break
# print("done")
# exit()

optimizer = torch.optim.Adam(train_params, lr=8e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


loss_log = {"nll": [],
            "val_nll": [],
            "epoch": [],
            "batch": []}

print("preparing run directory...")
os.mkdir(run_dir)
os.mkdir(os.path.join(run_dir, "checkpoints/"))
print("...done")


print("epoch \t batch \t nll \t aggregated nll \t validation nll")

for e in range(n_epochs):
    agg_nll = []
    for i, (im, cond) in enumerate(train_loader):

        # plt.imshow(im)
        # plt.show()

        source = cond[:, 0, :, :].to(device)
        target = cond[:, 1, :, :].to(device)

        # im = im.to(device)
        # cond = cond.to(device)

        out, log_j = cinn(target, c=source)

        alt_nll = torch.mean(out ** 2 / 2) - torch.mean(log_j) / ndim_total

        nll = 0.5 * torch.sum(out ** 2, dim=1) - log_j
        nll = torch.mean(nll) / ndim_total

        alt_nll.backward()

        # print("{}\t{}\t{}".format(e, i, alt_nll.item()))

        torch.nn.utils.clip_grad_norm_(train_params, 100.)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if i % 20 == 0:
            with torch.no_grad():
                val_x, val_c = next(iter(val_loader))
                val_s = val_c[:, 0, :, :].to(device)
                val_t = val_c[:, 1, :, :].to(device)
                # val_x, val_c = val_x.to(device), val_c.to(device)

                v_out, v_log_j = cinn(val_t, c=val_s)
                v_nll = torch.mean(v_out ** 2) / 2 - torch.mean(v_log_j) / ndim_total
                loss_log['nll'].append(alt_nll.item())
                loss_log['val_nll'].append(v_nll.item())
                loss_log['epoch'].append(e)
                loss_log["batch"].append(i)
                print("{}\t{}\t{}\t{}\t{}".format(e, i, alt_nll.item(), nll.item(), v_nll.item()))

        if i == 0:
            checkpoint = {
                "state_dict": cinn.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_{}_{}.pt'.format(e, i)))
    scheduler.step()

checkpoint = {
    "state_dict": cinn.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
}
torch.save(checkpoint, os.path.join(run_dir, 'checkpoints/model_final.pt'))
