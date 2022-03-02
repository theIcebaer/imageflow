import os
import sys
import itertools

import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


block_depth = [4, ]
data_res = (28, 28)

cond_channels = [(32, 64, 128),
                 (64, 128, 128),
                 (64, 128, 256)]
                 # (64, 128, 128, 256)]
conv_channels = {
    3: [(32, 64, 128),
        (64, 64, 128),
        (64, 128, 256)],
    4: [(32, 64, 128, 128),
        (32, 64, 128, 256),
        (64, 64, 128, 128)]
}
downsampling = {
    3: [(1, 0)],
    4: [(1, 0, 0),
        (0, 0, 1),
        (0, 1, 0)]
}

splits = {
    3: [
        ((2, 6), (4, 4)),
        ((4, 4), (2, 6)),
        ((4, 4), (4, 4))
        ],
    4: [
        ((2, 6), (4, 4), (8, 8)),
        ((4, 4), (2, 6), (8, 8)),
        ((4, 4), (4, 4), (8, 8)),
        ((2, 6), (2, 6), (8, 8))
        ]
}


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
    loss_log = {4: {cc: {s: {conv_c: {} for conv_c in conv_channels[len(cc)]} for s in splits[len(cc)]} for cc in cond_channels}}

    for bd in block_depth:
        for cc in cond_channels:
            for d in downsampling[len(cc)]:
                for s in splits[len(cc)]:
                    for conv_c in conv_channels[len(cc)]:
                        dir = f"architecture/supervised_{bd}_{cc}_{s}_{conv_c}/checkpoints/"

                        loss_log[bd][cc][s][conv_c] = torch.load(os.path.join(dir, "loss_log.pt"))

    return loss_log


# def test_models():
#     wrapper_test_data = [
#         ("mobilenet", (64, 960, 1, 1)),
#         ("resnet", (64, 512, 1, 1))
#     ]
#
#     pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']
#
#
#     for pretrain, (net_name, arch_params) in itertools.product(pretrain, wrapper_test_data):
#         dir = f"supervised_{net_name}_{pretrain}/checkpoints/"
#         model_dir = os.path.join(dir, "model_final.pt")
#         model = load_model()
#
#         for source, target in test_loader:
#             target_ = model(source)
#             err = np.mean(np.abs(target_ - target))



def make_tables(loss_log):


    tables = [f"\t & {conv_channels[len(cc)][0]} & {conv_channels[len(cc)][1]} & {conv_channels[len(cc)][2]} \\\\ \n" for i, cc in enumerate(cond_channels)]
    # idx = {md:i for i,md in enumerate(model_depth_values)}

    for i, s in enumerate(splits[3]):
        print(s)
        for cc in (cond_channels):
            line = f"{cc} "
            for conv_c in conv_channels[len(cc)]:
                # log = loss_log[sw][sd][md]
                x = round(loss_log[4][cc][s][conv_c]['nll'][-1], 2)
                line += f"& {x} "
            line += "\\\\ \n"
            tables[i] += line
    [(splits[3][i], print(t)) for i, t in enumerate(tables)]
    return tables

logs = load_logs()
# log_array = dict_to_np(logs)
# plot_losses_over_train(logs)
make_tables(logs)