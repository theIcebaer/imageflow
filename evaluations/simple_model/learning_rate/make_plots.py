import glob
import os
from multiprocessing import Process

from imageflow.tester import test_model
from imageflow.testDistribution import get_err_distribution

def draw_test_progression(err_list):
    import matplotlib.pyplot as plt
    import numpy as np

    for dir, err_measures in err_list.items():
        fig = plt.figure()
        means = np.array(err_measures['mean'])
        stds = np.array(err_measures['std'])

        plt.plot(means[:,0], label='mean') # 0 is the image index here since we do this per image
        plt.plot(stds[:,0], label='std')
        plt.legend()
        plt.show()
        # plt.plot(err_measures['error'])

err_list = {}

training_points = ['10_0', '20_0', '30_0', '40_0', '50_0', 'final']
for dir in glob.glob("Annealing*"):
    err_list[dir] = {
        'error': [],
        'mean': [],
        'std': []
    }

    for model_name in training_points:

        print("* ----------------------------")
        print(f" evaluating model with {dir}:")
        # l = dir.split("_")[2]
        # mult = dir.split("_")[3]
        # if mult is None:
        #     mult = 1.0
        # print(l, mult)


        model_dir = f"{dir}/checkpoints/model_{model_name}.pt"
        data_dir = "../../../data"
        if model_name == 'final':
            plot_dir = f"{dir}/plots"
        else:
            plot_dir = f"{dir}/plots_{model_name}"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        # else:
        #     print('skipping')
        #     continue

        kwargs = {}
        kwargs["model_dir"] = model_dir
        kwargs["data_dir"] = data_dir
        kwargs["model_type"] = "basic"
        kwargs["plot"] = True
        kwargs["plot_dir"] = plot_dir
        kwargs["data"] = "MNIST"
        kwargs["data_res"] = (28, 28)
        # kwargs["block_depth"] = 4
        # kwargs["cond_channels"] = (32, 64, 128)
        # kwargs["conv_channels"] = (64, 128, 256)
        # kwargs["splits"] = ((2, 6), (4, 4))
        # kwargs["downsample"] = (1, 0)
        kwargs["batch_size"] = 10
        kwargs["samples_per_datapoint"]=100
        kwargs["only_plot"] = True

        err_measures = get_err_distribution(**kwargs)
        err_list[dir]['mean'].append(err_measures['mean'])
        err_list[dir]['std'].append(err_measures['std'])
        err_list[dir]['error'].append(err_measures['error'])
        # reconstruction_err, field_err = test_model(**kwargs)
        # p = Process(target=test_model, kwargs=kwargs)
        # p.start()
        # p.join()

draw_test_progression(err_list)

