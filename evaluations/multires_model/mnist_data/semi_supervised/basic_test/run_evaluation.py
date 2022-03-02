import os
import itertools
from imageflow.trainer import train_unsupervised
from imageflow.trainer import train_supervised

data_dir = f"../../../../../data"

reg_params = {"penalty": "l1",
               "multiplier": 0.1}

for reg in [None, 'grad_loss']:

    pretrain_epochs = 60
    pretrain_run_dir = f"pretrain"
    if reg is not None:
        pretrain_run_dir += f"_{reg}"

    if os.path.isdir(pretrain_run_dir):
        print(f"Model already trained. Skipping supervised model: {pretrain_run_dir}")

    train_epochs = 60
    run_dir = f"model_{reg}"

    if os.path.isdir(run_dir):
        print(f"Model already trained. Skipping training unsupervised model: {pretrain_run_dir}")

    kwargs = {}
    kwargs["data"] = "MNIST"
    kwargs["model_type"] = "multiRes"
    kwargs[reg] = reg_params[reg]

    # pretrain model supervised
    pretrained_model, train_loss, val_loss = train_supervised(data_dir=data_dir, run_dir=pretrain_run_dir, n_epochs=pretrain_epochs, **kwargs)

    if reg is None:
        kwargs['grad_loss'] = reg_params[reg]

    train_unsupervised(data_dir=data_dir, run_dir=run_dir, checkpoint="pretrain/checkpoints/model_final.pt", n_epochs=train_epochs, **kwargs)

