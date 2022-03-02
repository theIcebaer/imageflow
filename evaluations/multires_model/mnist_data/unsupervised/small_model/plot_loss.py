import os
import glob
import matplotlib.pyplot as plt
from imageflow.utils import get_plot_data
losses = {dir: get_plot_data(os.path.join(dir, "loss.log"))[:2] for dir in glob.glob("*(8, 8, 8)*_60")}


for dir, (train, val) in losses.items():
    plt.plot(train)

plt.ylim(0,1)
plt.show()