import os
import itertools
from imageflow.trainer import train_unsupervised
from multiprocessing import Process

data_dir = f"../../../../../data"

regularizers = ['grad_loss', 'curl']

params = {
    'grad_loss': {
        "penalty": ["l1", "l2"],
        "multiplier": [0, 0.1, 0.3, 1]

    },

    'curl': {
        "multiplier": [10.0, 100.0, 1000.0]
    }
}
epochs = [10, 50]

for n in epochs:

    for r in regularizers:
        p = params[r]
        # print(p.keys())
        param_list =[p[k] for k in p.keys()]
        param_names = list(p.keys())
        # print(param_list)
        combs = list(itertools.product(*param_list))
        for comb in combs:


            param_string = "".join(f"_{c}" for c in comb)
            run_dir = f"{r}{param_string}"
            if n != 10:
                run_dir += f"_{n}"

            if os.path.isdir(run_dir):
                print("* Model ist already trained. Skipping to the next one.")
                continue

            reg_args = {param_names[i]: c for i, c in enumerate(comb)}
            print(reg_args)
            kwargs = {}

            kwargs[r] = reg_args
            kwargs["data"] = "MNIST"
            kwargs["model_type"] = "multiRes"
            kwargs["run_dir"] = run_dir
            kwargs["data_dir"] = data_dir
            kwargs["n_epochs"] = n
            print(kwargs)
            print("--")
            p = Process(target=train_unsupervised, kwargs=kwargs)
            p.start()
            p.join()
            # train_unsupervised(**kwargs)