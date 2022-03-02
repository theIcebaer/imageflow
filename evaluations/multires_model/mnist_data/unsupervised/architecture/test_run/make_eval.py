import os
from imageflow.tester import test_model

model_dir = "test_run/checkpoints/model_final.pt"
data_dir = "../../../../../data"

plot_dir = "test_run/plots"
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

reconstruction_err, field_err = test_model(model_dir=model_dir, data_dir=data_dir, model_type='multiRes')


print("image mean squared errors:")
[print(err) for err in reconstruction_err]

print("field mean squared errors:")
[print(err) for err in field_err]
