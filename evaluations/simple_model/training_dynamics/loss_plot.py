import glob
import os
from imageflow.visualize import join_plots_

models = ['supervised_xavier_1.0', 'supervised_xavier_0.1', 'supervised_xavier_0.001', 'supervised_gaussian_0.01', 'supervised_kaiming']
paths = [os.path.join(x, "checkpoints/loss_log.pt") for x in models] #glob.glob("supervised*")]

join_plots_(paths)