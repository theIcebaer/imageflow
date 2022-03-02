import os
import sys
import itertools

import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def dict_to_np(dic):
    data = list(dic.items())
    return np.array(data)


def plot_losses_over_train(loss_log):
    subnet_width_values = [128, 256, 512]
    subnet_depth_values = [1, 2, 3, 4, 5]
    model_depth_values = [2, 10, 20]
    # figs = {key: plt.figure(key) for key in model_depth_values}

    map = {}
    i = {key: 0 for key in model_depth_values}
    tables = {key: [] for key in model_depth_values}
    for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
        i[md] += 1
        # print(md)

        log = loss_log[sw][sd][md]
        plt.figure(md, figsize=(1, 2))
        plt.subplot(3, 5, i[md])
        plt.title(f"sw:{sw}, sd:{sd}, md:{md}")

        plt.plot(log['nll'], label='train loss')
        plt.plot(log['val_nll'], label='val loss')

        plot_min = min(log['nll'])
        plot_max = max(log['nll'])
        last = log['nll'][-1]
        print(f"{md} {sw} {sd}: {last}")
        plt.ylim(-2.5, 1.5)
        plt.hlines(xmin=0, xmax=50, y=plot_min, linestyles    ='dashed')

        plt.xlabel('epoch')
        plt.ylabel("nll")
        # plt.axis('equal')

        plt.legend()
    for md in model_depth_values:
        plt.figure(md)
        plt.tight_layout()
    plt.show()


def load_logs():
    subnet_width_values = [128, 256, 512]
    subnet_depth_values = [1, 2, 3, 4, 5]
    model_depth_values = [2, 10, 20]

    loss_log = {sw: {sd: {md: {} for md in model_depth_values} for sd in subnet_depth_values} for sw in subnet_width_values}

    for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
        dir = f"supervised_{sw}_{sd}_{md}/checkpoints/"

        loss_log[sw][sd][md] = torch.load(os.path.join(dir, "loss_log.pt"))

    return loss_log


def test_models():
    subnet_width_values = [128, 256, 512]
    subnet_depth_values = [1, 2, 3, 4, 5]
    model_depth_values = [2, 10, 20]

    loss_log = {sw: {sd: {md: {} for md in model_depth_values} for sd in subnet_depth_values} for sw in
                subnet_width_values}

    for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
        dir = f"supervised_{sw}_{sd}_{md}/checkpoints/"
        model_dir = os.path.join(dir, "model_final.pt")


def make_tables(loss_log):
    subnet_width_values = [128, 256, 512]
    subnet_depth_values = [1, 2, 3, 4, 5]
    model_depth_values = [2, 10, 20]
    tables = ["\t & 128 & 256 & 512 \\\\ \n" for _ in range(3)]
    idx = {md:i for i,md in enumerate(model_depth_values)}

    for md in model_depth_values:
        for sd in subnet_depth_values:
            line = f"{sd} "
            for sw in subnet_width_values:
                # log = loss_log[sw][sd][md]
                x = round(loss_log[sw][sd][md]['nll'][-1], 2)
                line += f"& {x} "
            line += "\\\\ \n"
            tables[idx[md]] += line
    [print(t) for t in tables]
    return tables

logs = load_logs()
# log_array = dict_to_np(logs)
# plot_losses_over_train(logs)
make_tables(logs)