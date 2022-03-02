import os
import matplotlib.pyplot as plt
os.environ['VXM_BACKEND'] = 'pytorch'

import voxelmorph as vxm
import imageflow.dataset
import numpy as np
import torch
from torch.nn.functional import pad

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# model = vxm.networks.VxmDense.load("/home/jens/thesis/imageflow/evaluations/voxelmorph/voxelmorph_model.pt", device=device)

# print("foo")
# def rewrite_file(path):
#     with open(path, 'r') as f:
#         log = f.read().split()
#         epoch = [log[0]]
#         batch = [log[1]]
#         loss = [log[2]]
#         log = log[3:]
#         print(len(log))
#         for i in range(len(log)/2):
#             e_len = 1 if e < 10 else 2
#
#
#
#

def show_loss(path):
    with open(path, 'r') as f:
        log = f.readlines()
        print(log)

    loss = [float(l.split()[2]) for l in log]
    # print(len(loss))

    plt.plot(loss)
    plt.show()

# show_loss("loss_log.txt")
# rewrite_file("/home/jens/thesis/imageflow/scripts/loss_log.txt")


def show_test_samples(model_path):
    model = vxm.networks.VxmDense.load(model_path, device="cpu")
    model.eval()

    from imageflow.dataset import CovidXDataset
    from torch.utils.data import DataLoader
    base_dir = "/home/jens/thesis/imageflow"
    # run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    data_dir = os.path.join(base_dir, "data")

    from imageflow.dataset import CovidXDataset
    data_set = CovidXDataset(data_dir)
    n_samples = len(data_set)
    rng = np.random.default_rng(42)
    indices = rng.permutation(np.arange(n_samples))
    n_test = int(n_samples * 0.8)
    test_indices = indices[n_test:]

    test_set = torch.utils.data.Subset(data_set, test_indices)

    test_loader = DataLoader(test_set, batch_size=8, drop_last=True)


    # field, cond = test_set[0]
    # source = cond[0][None, None,...]
    # target = cond[1][None, None,...]
    cond = next(iter(test_loader))
    source = cond[:, :1, ...]
    target = cond[:, 1:, ...]
    print(source.shape)
    print(target.shape)

    pred, p_field = model(source, target)

    fig = plt.figure()
    plt.imshow(pred.detach().numpy()[1, 0, ...])
    plt.show()
    # print(cond.shape)
    from imageflow.visualize import streamplot_from_batch, make_fig_from_batch
    streamplot_from_batch(p_field.detach().numpy(), show_image=False, get_all=True, save_path="plots/fields")
    make_fig_from_batch(pred.detach().numpy(), save_path="plots/prediction")


    # plt.imshow()

# show_test_samples("voxelmorph_model.pt")

def test_voxelmorph_model(model_path):
    model = vxm.networks.VxmDense.load(model_path, device="cpu")
    model.eval()

    from imageflow.dataset import CovidXDataset
    from torch.utils.data import DataLoader
    base_dir = "/home/jens/thesis/imageflow"
    # run_dir = os.path.join(base_dir, "runs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    data_dir = os.path.join(base_dir, "data")

    from imageflow.dataset import CovidXDataset
    data_set = CovidXDataset(data_dir)
    n_samples = len(data_set)
    rng = np.random.default_rng(42)
    indices = rng.permutation(np.arange(n_samples))
    n_test = int(n_samples * 0.8)
    test_indices = indices[n_test:]

    test_set = torch.utils.data.Subset(data_set, test_indices)

    test_loader = DataLoader(test_set, batch_size=20, drop_last=True)


    image_err = []
    field_err = []
    for i, cond in enumerate(test_loader):
        if i > 100:
            break
        source = cond[:, :1, ...]
        target = cond[:, 1:, ...]
        # source = pad(source, (2, 2, 2, 2))
        # target = pad(target, (2, 2, 2, 2))
        # print(source.shape)
        # print(target.shape)

        pred, p_field = model(source, target)
        image_err.append(torch.nn.functional.mse_loss(pred, target).detach().numpy())
        # field_err.append(torch.nn.functional.mse_loss(p_field[:, :, 2:30, 2:30], field[:, :, ::2, ::2]).detach().numpy())

    print(np.mean(image_err))
    print(np.mean(field_err))
    return image_err, field_err
# test_voxelmorph_model("/home/jens/thesis/imageflow/evaluations/voxelmorph/Covid/voxelmorph_model.pt")
show_test_samples("/home/jens/thesis/imageflow/evaluations/voxelmorph/Covid/voxelmorph_model.pt")