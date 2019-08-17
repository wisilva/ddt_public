# ddt_public
# Differentiable Decision Trees
Information regarding the flat ddt's hyperparameters is listed in the flat_ddt.py file.

All of the code for the DDT and its variations was written using Pytorch. Input data must be stored in a torch tensor of size batch_size x input_size

An object of the DDT class must be created and called iteratively over the training data, just like any torch.nn.module class.

Hyperparameters:

input_dim: int, number of input features, for example image pixels
output_size: int, number of possible labels
leaves: int, number of leaves, determines the depth of the tree. Must be 2^n, where n is the depth
alpha: double, temperature value, determines confidence of the model 
train_alpha: boolean, determines whether alpha is made a parameter of the model 
use_gpu: boolean, determines whether the model runs on the gpu to decrease processing time. In order to use, input data must be stored on a gpu
vectorized: boolean, if true, comparators of probability nodes are vectorized with output determined by attention layer 
is_value: boolean, if false, runs output through softmax before returning it, set to true if user optimizers like Adam that already use softmax

net = DeepDDT(input_dim=30, leaves=16, output_dim=2, alpha=1.0, use_gpu=True, is_value=True, vectorized=True, train_alpha=False)

result = net(input_data)
