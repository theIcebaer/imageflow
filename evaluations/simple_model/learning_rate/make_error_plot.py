import imageflow.testDistribution as dist
import os
import glob

training_points = ['10_0', '20_0', '30_0', '40_0', '50_0', 'final']

for model_name in training_points:
    for dir in glob.glob("Annealing*"):
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

        # else:
        #     print('skipping')
        #     continue

        kwargs = {}
        kwargs["model_dir"] = model_dir
        kwargs["data_dir"] = data_dir
        kwargs["model_type"] = "basic"

    dist.get_err_distribution(**kwargs)