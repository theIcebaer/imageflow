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
    wrapper_test_data = [
        ("mobilenet", (64, 960, 1, 1)),
        ("resnet", (64, 512, 1, 1))
    ]

    pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']
    map = {}

    i = {key: 0 for key,_ in wrapper_test_data}
    # tables = {name: [] for name, _ in wrapper_test_data}

    for pretrain, (net_name, arch_params) in itertools.product(pretrain, wrapper_test_data):
        i[net_name] += 1
        # print(md)

        log = loss_log[net_name][pretrain]

        plt.figure(i[net_name])
        # plt.subplot(3, 5, i[md])
        plt.title(f"{net_name}: {pretrain}")

        plt.plot(log['nll'], label='train loss')
        plt.plot(log['val_nll'], label='val loss')

        plot_min = min(log['nll'])
        plot_max = max(log['nll'])
        last = log['nll'][-1]
        # print(f"{md} {sw} {sd}: {last}")
        plt.ylim(-2.5, 1.5)
        plt.hlines(xmin=0, xmax=50, y=plot_min, linestyles    ='dashed')

        plt.xlabel('epoch')
        plt.ylabel("nll")
        # plt.axis('equal')

        plt.legend()

        plt.show()


def load_logs():
    wrapper_test_data = [
        ("mobilenet", (64, 960, 1, 1)),
        ("resnet", (64, 512, 1, 1))
    ]

    pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']

    loss_log = {net: {pt: {} for pt in pretrain} for net, _ in wrapper_test_data}

    for pretrain, (net_name, arch_params) in itertools.product(pretrain, wrapper_test_data):
        dir = f"supervised_{net_name}_{pretrain}/checkpoints/"

        loss_log[net_name][pretrain] = torch.load(os.path.join(dir, "loss_log.pt"))

    return loss_log


def test_models():
    wrapper_test_data = [
        ("mobilenet", (64, 960, 1, 1)),
        ("resnet", (64, 512, 1, 1))
    ]

    pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']


    for pretrain, (net_name, arch_params) in itertools.product(pretrain, wrapper_test_data):
        dir = f"supervised_{net_name}_{pretrain}/checkpoints/"
        model_dir = os.path.join(dir, "model_final.pt")
        model = load_model()

        for source, target in test_loader:
            target_ = model(source)
            err = np.mean(np.abs(target_ - target))



def make_tables(loss_log):
    wrapper_test_data = [
        ("mobilenet", (64, 960, 1, 1)),
        ("resnet", (64, 512, 1, 1))
    ]

    pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']



    table = "\t & fixed & tuning on last layers & tuning on whole network & no pretraining \\\\ \n"
    # idx = {md:i for i,md in enumerate(model_depth_values)}

    for net_name, _ in wrapper_test_data:
        line = f"{net_name} "
        for pt in pretrain:
            x = round(loss_log[net_name][pt]['nll'][-1], 2)
            line += f"& {x} "
        line += "\\\\ \n"
        table += line
    print(table)
    return table

logs = load_logs()
# log_array = dict_to_np(logs)
plot_losses_over_train(logs)
# make_tables(logs)