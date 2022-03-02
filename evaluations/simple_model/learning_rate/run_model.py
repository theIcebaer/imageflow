import os.path

from imageflow.trainer import train_supervised

schedulers = ['Annealing', "AnnealingWarmRestarts", "MultiStep"]
Tmax = [10]
T0 = [10, 10, 60, 9]
Tmult = [1, 2, 1, 2]
weight_decay = [0.2, 0.1, 0.1, 0.01 ]
milestones = [[20, 40], [20,40], [20, 30, 40, 50], [10, 20, 30, 40, 50]]

cosine_parameters = {
    "Annealing": [{'T_max': Tmax[i]} for i in range(len(Tmax))],
    "AnnealingWarmRestarts": [{"T_0": T0[i], "T_mult":Tmult[i]} for i in range(len(T0))],
    "MultiStep": [{'milestones': milestones[i], 'gamma': weight_decay[i]} for i in range(4)]
}

for sched in schedulers:
    for param in cosine_parameters[sched]:
        data_dir = "/home/jens/thesis/imageflow/data"

        run_dir = f"{sched}"

        for k,v in param.items():
            run_dir += f"_{v}"

        if os.path.isdir(run_dir):
            print("already done. skipping.")
            continue

        kwargs = {}
        kwargs['model_type'] = 'basic'
        kwargs['data_dir'] = data_dir
        kwargs['run_dir'] = run_dir
        kwargs['n_epochs'] = 60
        kwargs['scheduler'] = sched
        kwargs['scheduler_params'] = param

        train_supervised(**kwargs)