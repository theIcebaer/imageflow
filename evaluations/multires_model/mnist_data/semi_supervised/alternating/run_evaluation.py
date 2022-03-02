import os
import itertools
from imageflow.trainer import train_unsupervised
from imageflow.trainer import train_supervised

data_dir = f"../../../../../data"


pretrain_epochs = 2
pretrain_run_dir = "pretrain"

train_epochs = 5
run_dir = "model"

kwargs = {}
kwargs["data"] = "MNIST"
kwargs["model_type"] = "multiRes"


# pretrain model supervised
pretrained_model, train_loss, val_loss = train_supervised(data_dir=data_dir, run_dir=pretrain_run_dir, n_epochs=pretrain_epochs, **kwargs)

kwargs["grad_loss"] = {"penalty": "l1",
                       "multiplier": 0.1}

train_unsupervised(data_dir=data_dir, run_dir=run_dir, pretrained_model=pretrained_model, n_epochs=train_epochs, **kwargs)



# for r in regularizers:
#     p = params[r]
#     # print(p.keys())
#     param_list =[p[k] for k in p.keys()]
#     param_names = list(p.keys())
#     # print(param_list)
#     combs = list(itertools.product(*param_list))
#     for comb in combs:
#
#
#         param_string = "".join(f"_{c}" for c in comb)
#         run_dir = f"{r}{param_string}"
#
#         if os.path.isdir(run_dir):
#             print("* Model ist already trained. Skipping to the next one.")
#             continue
#
#         reg_args = {param_names[i]: c for i, c in enumerate(comb)}
#
#         kwargs = {}
#
#         kwargs[r] = reg_args
#         kwargs["data"] = "MNIST"
#         kwargs["model_type"] = "multiRes"
#         kwargs["run_dir"] = run_dir
#         kwargs["data_dir"] = data_dir
#         print(kwargs)
#         print("--")
#
#         train_unsupervised(**kwargs)