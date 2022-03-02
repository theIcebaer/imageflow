import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from imageflow.nets import Reg_mnist_cINN
from imageflow.nets import CinnBasic
from imageflow.nets import CinnConvMultiRes
from imageflow.dataset import MnistDataset
from imageflow.utils import apply_flow
from torch.nn.functional import mse_loss as mse
from utils import plot_results


# -- load model ----
data_dir = '../data'
model_dir = '../runs/2021-10-28_15-22/checkpoints/model_final.pt'
batch_size = 10
ndim_total = 28 * 28 * 2
plot = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading model...")
model_dict = torch.load(model_dir)
state_dict = {k: v for k, v in model_dict['state_dict'].items() if 'tmp_var' not in k}
cinn = CinnBasic(device=device)
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


# -- generate samples ----
n_samples = 10
v_field, cond = next(iter(test_loader))
cond = cond.to(device)
v_fields = []

for _ in range(n_samples):
    z = torch.randn(batch_size, ndim_total).to(device)
    target_pred, v_field_pred, log_jac = cinn.reverse_sample(z, cond)
    v_fields.append(v_field_pred[0].to(torch.device('cpu')).detach().numpy())

    source = cond[:, :1, ...].to(device)
    target = cond[:, 1:, ...].to(device)

    # plot_results(source, target, v_field, target_pred, v_field_pred, index=0)


# -- make distance matricies ----
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

v_fields = np.array(v_fields).reshape(10, 2*28*28)
tsne = TSNE(n_components=2, learning_rate='auto', init='random')
pca = PCA()
# kernelPca =
# v_fields_embedded = tsne.fit_transform(v_fields)
v_fields_pca = pca.fit_transform(v_fields)
pass

# -- cluster samples -----


# -- reduce dimensions & evaluate t-sne ----

