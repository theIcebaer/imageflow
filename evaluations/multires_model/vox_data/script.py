import numpy as np

npz = np.load('/home/jens/thesis/imageflow/data/brains/tutorial_data.npz')
x_train = npz['train']
x_val = npz['validate']

print(x_train.shape)
print(x_val.shape)

vol_shape = x_train.shape[1:]
print('train shape:', x_train.shape)
