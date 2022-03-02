import os
from imageflow.tester import test_model
import glob
import numpy as np




# im_errors = {x: {y: 0 for y in [0.1, 0.5, 0.9, 1.0]} for x in ["l1", "l2"]}
# field_errors = {x: {y: 0 for y in [0.1, 0.5, 0.9, 1.0]} for x in ["l1", "l2"]}
for dir in glob.glob("*"):
    print("* ----------------------------")
    print(f" evaluating model with {dir}:")
    # l = dir.split("_")[2]
    # mult = dir.split("_")[3]
    # if mult is None:
    #     mult = 1.0
    # print(l, mult)


    model_dir = f"{dir}/model_final.pt"
    data_dir = "../../../data"

    plot_dir = f"{dir}/plots"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    kwargs = {}
    kwargs["data_res"] = (28, 28)
    # kwargs["block_depth"] = 4
    # kwargs["cond_channels"] = (32, 64, 128)
    # kwargs["conv_channels"] = (64, 128, 256)
    # kwargs["splits"] = ((2, 6), (4, 4))
    # kwargs["downsample"] = (1, 0)
    kwargs["batch_size"] = 10


    reconstruction_err, field_err = test_model(model_dir=model_dir, data_dir=data_dir, model_type='basic', plot=False, plot_dir=plot_dir, data="MNIST", **kwargs)
    print(reconstruction_err)
    print(field_err)
    im_test_err = np.mean(reconstruction_err)
    field_test_err = np.mean(field_err)
    print(f"reconstruction error{reconstruction_err}")
    print(f"field error: {field_test_err}")
    # im_errors[l][mult] = round(im_test_err, 3)
    # field_errors[l][mult] = round(field_test_err, 3)
    # print(field_errors[l][mult])
    # print(im_errors[l][mult])
# im_table = ["l1 & l2 \\\\ \n"]
# field_table = ["l1 & l2\\\\ \n"]
#
# for m in [0.1, 0.5, 0.9, 1.0]:
#     im_line = f"{m} "
#     f_line = f"{m} "
#     for l in ["l1", "l2"]:
#         im_line += f"& {im_errors[l][m]}"
#         f_line += f"& {field_errors[l][m]}"
#     print(f_line)
#     im_table.append(f"{im_line} \\\\ \n")
#     field_table.append(f"{f_line} \\\\ \n")
#
# print(field_table)
#
# with open("table_smooth_reg_ims.txt", "w") as f:
#     f.writelines(im_table)
#
# with open("table_smooth_reg_field.txt", "w") as f:
#     f.writelines(field_table)
#
#
# print("image mean squared errors:")
# [print(err) for err in reconstruction_err]
#
# print("field mean squared errors:")
# [print(err) for err in field_err]

