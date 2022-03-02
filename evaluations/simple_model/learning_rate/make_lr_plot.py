import os
import glob
import torch
import matplotlib.pyplot as plt


fig = plt.figure()





logs = []
for dir in glob.glob("Annea*"):
    path = os.path.join(dir, 'checkpoints/loss_log.pt')
    dir_, filename = os.path.split(path)
    if filename.split(".")[-1] == 'pt':
        name = dir_.split('/')[-2].split('_')
        type = name[0]
        if type == "AnnealingWarmRestarts":
            type = "Annealing Warm Restarts"
            if name[-1] == str(2):
                type+= "-dampened"
        loss_log = torch.load(path)
        logs.append(loss_log)
        plt.plot(loss_log['nll'], label=f'{type}')
        # plt.plot(loss_log['val_nll'], label='{} validation loss')
        # make tiks



    else:
        raise AttributeError("loss log format not implemented yet.")

n_datapoints = len(logs[0]['nll'])
ticks = [int(n_datapoints / 4), int(n_datapoints / 2), int(n_datapoints * 3 / 4), n_datapoints]
labels = [logs[0]['epoch'][x - 1] + 1 for x in ticks]
plt.xticks(ticks, labels)
plt.ylim(-2, 2)
plt.xlabel("epoch")
plt.ylabel(r"$\mathcal{L}_{nll}$")
plt.legend()
plt.savefig("lr_plot")
plt.show()



