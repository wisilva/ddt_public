# Differentiable Decision Trees

This repo houses all of the code for the Deep DDT and its variations. It is written using [PyTorch](https://pytorch.org/get-started/locally/) and [NumPy](https://www.numpy.org/). In order to use these classes, you must have PyTorch and numpy in your environment, but that's it. For more information on what a DDT is and why it might be useful, see the DDT write-up pdf.

Input data must be in a torch tensor of size `batch_size x input_size` where `batch_size` is the number of samples going in at once, and `input_size` is the dimension of the input.

Using one of the DDT classes works the same as any other PyTorch network class or `nn.Module`, so it's use should be familiar.

### Hyperparameters
There are a few hyperparameters that must be set for each instantiation of a DDT. They are:

* `input_dim`: int, number of input features
* `output_size`: int, number of possible labels
* `leaves`: int, number of leaves. This determines the depth of the tree. Must be 2^n, where n is the depth
* `alpha`: float, temperature value. This determines confidence of the model at decision nodes.
* `train_alpha`: Boolean flag. This determines whether alpha is made a parameter of the model.
* `use_gpu`: Boolean flag. This determines whether the model runs on the GPU. In order to use, input data must be placed on a GPU.
* `vectorized`: Boolean flag. If true, comparators of probability nodes are vectorized with output determined by attention layer
* `is_value`: Boolean flag. If false, runs output through softmax before returning it.

For a 30 dimension input with 2 output classes, a static alpha = 1.0, a vectorized DDT that doesn't softmax output, on the GPU, a Deep DDT model would be created like so:

```
net = DeepDDT(input_dim=30, leaves=16, output_dim=2, alpha=1.0, use_gpu=True, is_value=True, vectorized=True, train_alpha=False)
```

And to get a prediction from the network, as in any other `nn.Module` in PyTorch, simply create your input_data tensor and run:
```
result = net(input_data)
```
