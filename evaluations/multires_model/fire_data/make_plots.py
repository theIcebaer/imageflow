import glob
import os

from imageflow.tester import test_model

for dir in glob.glob("grad*"):
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
    kwargs["data_res"] = (184, 184)
    # kwargs["block_depth"] = 4
    # kwargs["cond_channels"] = (32, 64, 128)
    # kwargs["conv_channels"] = (64, 128, 256)
    # kwargs["splits"] = ((2, 6), (4, 4))
    # kwargs["downsample"] = (1, 0)
    kwargs["batch_size"] = 8
    kwargs["data_res"] = (184, 184)
    kwargs["block_depth"] = 2
    kwargs["cond_channels"] = (32, 32, 32)
    kwargs["conv_channels"] = (32, 32, 32)



    reconstruction_err, field_err = test_model(model_dir=model_dir, data_dir=data_dir, model_type='multiRes', plot=True, plot_dir=plot_dir, data="FIRE", **kwargs)