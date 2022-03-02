import os
from imageflow.tester import test_model
import glob
import numpy as np
import torch

im_errors = {x: {y: 0 for y in [0.1, 0.5, 0.9, 1.0]} for x in ["l1", "l2"]}
field_errors = {x: {y: 0 for y in [0.1, 0.5, 0.9, 1.0]} for x in ["l1", "l2"]}
for dir in glob.glob("supervised_4_(64, 128, 256)*"):
    print("* ----------------------------")
    print(f" evaluating model with {dir}:")

    params = dir.split("/")[0].split("_")
    bd = int(params[1])
    cc = [int(x.strip('(, )')) for x in params[2].split(",")]
    conv_c = [int(x.strip('(, )')) for x in params[4].split(",")]
    s = [int(x.strip('(, )')) for x in params[3].split(",")]
    s = ((s[0], s[1]), (s[2], s[3]))
    d = (1, 0)

    # l = dir.split("_")[2]
    # mult = dir.split("_")[3]
    # if mult is None:
    #     mult = 1.0
    # # print(l, mult)

    # method = dir.split("_")[0]
    # mult = dir.split("_")[-1]
    # for model in ["model_final", "model_2_180"]:
    #     model_dir = f"{dir}/checkpoints/{model}.pt"
    model_dir = f"{dir}/checkpoints/model_final.pt"
    # model_dir = f"{dir}/checkpoints/model_2_180.pt"
    data_dir = "../../../../../data"

    plot_dir = f"{dir}/plots_{model_dir.split('/')[-1]}"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    reconstruction_err, field_err = test_model(model_dir=model_dir,
                                               data_dir=data_dir,
                                               model_type='multiRes',
                                               plot=True,
                                               plot_dir=plot_dir,
                                               index=torch.arange(10),
                                               batch_size=10,
                                               block_depth=bd,
                                               cond_channels=cc,
                                               splits=s,
                                               downsample=d,
                                               conv_channels=conv_c,
                                               stream=True)
    print(reconstruction_err)
    print(field_err)
    im_test_err = np.mean(reconstruction_err)
    field_test_err = np.mean(field_err)

    im_errors[dir] = round(im_test_err, 3)
    field_errors[dir] = round(field_test_err, 3)
    print(field_errors[dir])
    print(im_errors[dir])
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
# #
# #
# # print("image mean squared errors:")
# # [print(err) for err in reconstruction_err]
# #
# # print("field mean squared errors:")
# # [print(err) for err in field_err]
