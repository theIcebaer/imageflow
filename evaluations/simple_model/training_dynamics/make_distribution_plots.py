import glob
import os

from imageflow.tester import test_model
from imageflow.testDistribution import get_err_distribution

fields_list = []

for dir in glob.glob("supervised*"):
    print("* ----------------------------")
    print(f" evaluating model with {dir}:")
    # l = dir.split("_")[2]
    # mult = dir.split("_")[3]
    # if mult is None:
    #     mult = 1.0
    # print(l, mult)


    model_dir = f"{dir}/checkpoints/model_final.pt"
    data_dir = "../../../data"

    plot_dir = f"{dir}/plots"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    else:
        print('skipping')
        continue

    kwargs = {}
    kwargs["data_res"] = (28, 28)
    # kwargs["block_depth"] = 4
    # kwargs["cond_channels"] = (32, 64, 128)
    # kwargs["conv_channels"] = (64, 128, 256)
    # kwargs["splits"] = ((2, 6), (4, 4))
    # kwargs["downsample"] = (1, 0)
    kwargs["batch_size"] = 10


    fields = get_err_distribution(model_dir=model_dir, data_dir=data_dir, model_type='basic', plot=False, plot_dir=plot_dir, data="MNIST", **kwargs)

