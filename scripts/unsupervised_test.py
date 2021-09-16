import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from imageflow.nets import Reg_mnist_cINN
from imageflow.dataset import MnistDataset
from imageflow.utils import apply_flow
from torch.nn.functional import mse_loss as mse

data_dir = 'data'
model_dir = 'runs/2021-09-15_20-28/checkpoints/model_final.pt'
batch_size = 256
ndim_total = 28 * 28 * 2
plot = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading model...")
model_dict = torch.load(model_dir)
state_dict = {k: v for k, v in model_dict['state_dict'].items() if 'tmp_var' not in k}
cinn = Reg_mnist_cINN()
cinn.to(device)
cinn.load_state_dict(state_dict)
print("...done")

print("Loading data...")
data_set = MnistDataset(os.path.join(data_dir, "mnist_rnd_distortions_1.hdf5"))
train_set, val_set, test_set = torch.utils.data.random_split(data_set, [47712, 4096, 8192],
                                                             generator=torch.Generator().manual_seed(42))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True)
print("...done")

cinn.eval()

with torch.no_grad():
    field_err = []
    reconstruction_err = []

    for v_field, c in test_loader:

        v_field = v_field.to(device)
        # print(v_field.shape)
        c = c.to(device)
        # x = true field
        # c = condition vector of composed moving and fixed image (m,f)
        z = torch.randn(batch_size, 2 * 28 * 28).to(device)
        v_field_pred, _ = cinn(z, c=c, rev=True)  # predicted velocity field

        source_imgs = c[:, 0, ...]

        source = c[:, :1, ...].to(device)
        target = c[:, 1:, ...].to(device)

        target_pred, _ = apply_flow(v_field=v_field_pred, volume=source, device=device)

        field_err.append(mse(v_field, v_field_pred).item())
        reconstruction_err.append(mse(target, target_pred).item())

        if plot:
            # Shift everything to cpu since you can't plot from gpu directly.
            source = source.to(torch.device("cpu"))
            target = target.to(torch.device("cpu"))
            target_pred = target_pred.to(torch.device("cpu"))
            v_field = v_field.to(torch.device("cpu"))
            v_field_pred = v_field_pred.to(torch.device("cpu"))

            # print(v_field.shape)
            # print(v_field_pred.shape)

            fig = plt.figure()

            idx = np.random.randint(source.shape[1])

            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title("source img")
            ax1.imshow(source[idx, 0, :, :])

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title("predictet target")
            ax2.imshow(target_pred[idx, 0, :, :])

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title("true target")
            ax3.imshow(target[idx, 0, :, :])
            ax3.set_xlabel("mse: {}".format(round(reconstruction_err[-1], 2)))

            fig2 = plt.figure()

            ax4 = fig2.add_subplot(1, 2, 1)
            ax4.set_title("predicted velocity field")
            # ax4.imshow(v_field_pred[idx, 0, :, :])
            u, v = v_field_pred[idx, 0, :, :], v_field_pred[idx, 1, :, :]
            x, y = np.meshgrid(np.linspace(0, 27, 28), np.linspace(0, 27, 28))
            plt.axis("equal")
            plt.streamplot(x, y, u, v)

            ax5 = fig2.add_subplot(1, 2, 2)
            ax5.set_title("true velocity field")
            # ax5.imshow(v_field[idx, 0, :, :])
            u, v = v_field[idx, 0], v_field[idx, 1]
            print(u.shape)
            x, y = np.meshgrid(np.linspace(-13, 14, 28), np.linspace(-13, 14, 28))
            plt.axis("equal")
            plt.streamplot(x, y, u, v)
            ax5.set_xlabel("mse: {}".format(round(field_err[-1], 2)))

            plt.show()