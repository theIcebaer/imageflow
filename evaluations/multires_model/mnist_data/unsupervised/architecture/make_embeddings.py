from imageflow.tester import make_embeddings

model_path = "/home/jens/thesis/imageflow/evaluations/multires_model/mnist_data/unsupervised/architecture/grad_loss_l2_1_50/checkpoints/model_final.pt"
# model_path = "/home/jens/thesis/imageflow/evaluations/multires_model/mnist_data/unsupervised/architecture/grad_loss_l2_0_50/checkpoints/model_10.pt"

make_embeddings(model_path, model_type="multiRes", method='mds', idx=[0], show_sample=False)