from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
#For a different DDT, change the file name and possibly the name of the DDT Class, ex "from flat_ddt import FlatDDT"
from deep_ddt import DeepDDT
"""
This file is an example of the use of the deep ddt in classification, specifically for the Breast Cancer Dataset
The Breast Cancer Data set is taken from the sklearn datasets. It contains 30 hand-selected input features, which describe
a cell nucleus sample, and 2 output labels, which indicate the diagnosis of the cell: benign or malignant.
"""

"""
The random seeds for pytorch and numpy are set so that the parameters of the model are always initialized in the same way
Although this should not matter overall, it allows for a consistent accuracy to be found so that the effects of changes to the model/varying hyperparameters can be established"""
np.random.seed(0)
torch.random.manual_seed(0)


"""
Here I've placed variables for most of the hyperparameters/model's attributes so that I can change them easily. 
Most of these are used in the creation of the DeepDDT object. The documentation for the Deep and Flat DDT describe the purpose of most of these.
"""
depth = 2
epochs = 300
train_alpha = False
temp = 1
vectorized = True
batch_size = 16
learning_rate = 8e-4




"""Instantiating the DDT Model.
 1. The input and output dimensions, in this case 30 and 2, must be specified for the model as "input_dim" and "output_dim"
 
 2. Because the current Deep DDT model (as well as the Flat DDT) does not use expert
initialization, set the nodes to None and set the leaves to '2**depth'. The nodes will be randomly initialized. 
Other hyperparameters can be found in the README for the Deep DDT and in comments in the Flat DDT's file

"""
net = DeepDDT(30,nodes = None, leaves = 2**depth,output_dim=2,alpha =temp, is_value=True, vectorized=vectorized, train_alpha=train_alpha)


"""
The loss function and optimizer that are called after each forward call must be initialized before. Because
Adam is being used as an optimizer, is_value must be set to True (or else the softmax function will be used twice)
The learning rate specified above is used here"""
lossCalc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

"""Loading the breast cancer data set. Each row of the contains a set of 30 input features and a single label, which represents
the true classification of the input"""
bc= load_breast_cancer()

"""This is a useful way to break sklearn datasets into separate arrays of the 30 features and the labels. The same data 
entry has corresponding indices for each of the arrays, so X[0] and y[0] refer to the features and label of the same input.
The input features are normalized, which generally speeds up the model's convergence"""
X = bc.data
X = preprocessing.normalize(X)
y = bc.target

"""Breaking up the dataset into a trainset and test set. 30% of the data will not be used for training so that an
accurate approximation of its performance can be found (testing using train data is deceptive to the model's actual
accuracy because the model could simply be 'memorizing' the data)."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


"""I have a running tally for the best accuracy so that I don't have to look for it. All code referring to this
can be removed for whatever reason."""
best = 0.0

"""This nested for loop is consistent for almost any model that actively trains and almost any data set. For the 
number of epochs, the data is broken into small batches and run through the forward method of the DDT."""

for epoch in range(epochs):
    for i in range(0, len(X_train),batch_size):

        """Break up the training data into batch sized "samples (X)" and "labels (y)" """
        sample = X_train[i:i+batch_size]
        label = y_train[i:i+batch_size]

        """For pytorch models, the input must be stored in a tensor with requires_grad set to true for backpropogation.
        I used Tensor for the inputs and LongTensor for the labels simply because that was what worked. 
        (requires_grad = True is the default)
        
        NOTE: This storage in a tensor does not need to be done at every iteration. The whole thing could be placed inside a tensor before 
        the loop begins, but I am keeping this inefficient solution in this file because I know it works. 
        It can be changed, which might (very) slightly improve the speed of training.
        
        NOTE: torch.autograd.Variable is apparently deprecated and no longer necessary, but it does not cause any problems."""
        sample = torch.autograd.Variable(torch.Tensor(sample))
        label = torch.autograd.Variable(torch.LongTensor(label))

        # Zero the optimizer, for the previous iteration's gradient is still stored
        optimizer.zero_grad()

        #Runs the input through the model and sets the Tensor of a batch_size number of labels to the "output variable"
        outputs = net(sample)

        #This calculates loss between the outputs and the actual labels and calculates the gradient
        loss = lossCalc(outputs, label)
        loss.backward()

        #Changes parameters of the model by the specified gradient descent, in this case Adam, and the given learning rate
        optimizer.step()

   #Training the remaining data as one batch if the dataset size is not a multiple of the batch size; this repeats the previous lines of code
    sample = X_train[i:]
    label = y_train[i:]
    sample = torch.autograd.Variable(torch.Tensor(sample))
    label = torch.autograd.Variable(torch.LongTensor(label))

    optimizer.zero_grad()

    outputs = net(sample)
    loss = lossCalc(outputs, label)
    loss.backward()
    optimizer.step()

    """Uses accuracy_score from sklearn to calcuate the percent accuracy of the model's classification.
    All of the test data (test_x) partitioned in the train_test_split is run through the model at once and matched with 
    the corresponding test labels. Instead of a probability distribution, the value that is taken from the output is
     the index of whichever output has the highest value, i.e. the "best guess" of the DDT. This is the argmax method.
     The label accuracy is then calculated as a decimal (which I turn into a percent)
     
     NOTE: This accuracy_score method can be replaced with an sklearn loss calculation. In this case, remove the multiplication by 100
     and look for the lowest loss value for the "best." """
    testx = torch.tensor(X_test, dtype=torch.float, requires_grad=True)

    res = net(torch.autograd.Variable(testx))
    #Stores the data of the output Tensor in a numpy array, as these are taken as inputs in accuracy score
    resAr = res.data.numpy()
    #Index of highest probability value
    y_pred = np.argmax(resAr, 1)

    print('Epoch #', epoch+1, (accuracy_score(y_test, y_pred) * 100))

    #Updates the 'best' value if the current epoch's accuracy value is higher. For loss, change to check for lowest value instead
    if((accuracy_score(y_test, y_pred) * 100)>best):
        best = (accuracy_score(y_test, y_pred) * 100)

print('Finished')

#Testing the accuracy of the train set. NOTE: It will not necessarily be higher than the test set's
testx = torch.tensor(X_test, dtype = torch.float, requires_grad=True)
res = net(torch.autograd.Variable(testx))

resAr = res.data.numpy()

y_pred = np.argmax(resAr, 1)
print("Test Set:",(accuracy_score(y_test, y_pred)*100))

#This is the same accuracy function as above called again. It should return the same value as the last epoch and can be removed
trainx  = torch.tensor(X_train, dtype = torch.float, requires_grad=True)
res = net(torch.autograd.Variable(trainx))

resAr = res.data.numpy()

y_pred = np.argmax(resAr, 1)

print("Train set:",(accuracy_score(y_train, y_pred)*100))
#Print statements to remind the user of the hyperparameters for this run as well has the highest accuracy.
print("Best:", best)
print('Alpha:', net.alpha, '| Depth:', depth,'| Learning Rate:', learning_rate)

"""Other notes
To run on the GPU, input data must be stored in a tensor with a GPU device specified (for example using CUDA)
It is not done here because BC is a relatively small data set and convergence is relatively fast already."""
