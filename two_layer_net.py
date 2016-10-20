
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.


import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent 
# instances of our network. The network parameters are stored in the instance variable `self.params` 
# where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and 
# a toy model that we will use to develop your implementation.

# In[ ]:

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

net = init_toy_model()
X, y = init_toy_data()


# Forward pass: compute scores
# Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. 
# This function is very similar to the loss functions you have written for the SVM and Softmax exercises: 
# It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 
# 
# Implement the first part of the forward pass which uses the weights and biases to compute the scores for all 
# inputs.

scores = net.loss(X)
print 'Your scores:'
print scores
print
print 'correct scores:'
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print correct_scores
print

# The difference should be very small. We get < 1e-7
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))


from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.1)

# these should all be less than 1e-8 or so
for param_name in grads:
  f = lambda W: net.loss(X, y, reg=0.1)[0]
  param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
  print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))


# # Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` and fill in the missing sections to implement the training procedure. 
# This should be very similar to the training procedure you used for the SVM and Softmax classifiers. You will also have to implement `TwoLayerNet.predict`, as the training process periodically performs prediction to keep track of accuracy over time while the network trains.
# 
# Once you have implemented the method, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.2.

net = init_toy_model()
stats, test_net = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iters=100, verbose=False)

print 'Final training loss: ', stats['loss_history'][-1]

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
# plt.show()

# # # Load the data
# # Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


# # Train a network
# To train our network we will use SGD with momentum. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

# In[ ]:

# input_size = 32 * 32 * 3
# hidden_size = 50
# num_classes = 10
# net = TwoLayerNet(input_size, hidden_size, num_classes)








# # Train the network
# stats = net.train(X_train, y_train, X_val, y_val, num_iters=10000, batch_size=100, learning_rate=1e-3, learning_rate_decay=0.95,
#             reg=0.5, verbose=True)

# # Predict on the validation set
# val_acc = (net.predict(X_val) == y_val).mean()
# print 'Validation accuracy: ', val_acc




# # In[ ]:


from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

# def show_net_weights(net):
#   W1 = net.params['W1']
#   W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
#   plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
#   plt.gca().axis('off')
#   plt.show()

# show_net_weights(net)

print "pause"

# # Tune your hyperparameters
# 
# **What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more 
# or less linearly, which seems to suggest that the learning rate may be too low. Moreover, 
# there is no gap between the training and validation accuracy, suggesting that the model we 
# used has low capacity, and that we should increase its size. On the other hand, with a very large 
# model we would expect to see more overfitting, which would manifest itself as a very large gap 
# between the training and validation accuracy.
# 
# **Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final 
# performance is a large part of using Neural Networks, so we want you to get a lot of practice. 
# Below, you should experiment with different values of the various hyperparameters, including 
# hidden layer size, learning rate, numer of training epochs, and regularization strength. 
# You might also consider tuning the learning rate decay, but you should be able to get good 
# performance using the default value.

# Ideas: PCA, Dropout, adding features


input_size = 32 * 32 * 3
num_classes = 10

lrates = [0.001]
regs = [0.02]
hidden_sizes = [100]

best_accuracy = 0
for lrate in lrates:
  for reg in regs:
    for hidden_size in hidden_sizes: 
      # Train the network with the combination
      net = TwoLayerNet(input_size, hidden_size, num_classes)
      stats, test_net = net.train(X_train, y_train, X_val, y_val, num_iters=10000, batch_size=200, learning_rate=lrate, learning_rate_decay = .95,
                  reg=reg, verbose=True)

      if stats['val_acc_history'][-1] > best_accuracy:
        best_net = test_net
        best_accuracy = stats['val_acc_history'][-1]
        best_loss = np.mean(stats["loss_history"][-10:-1])
        best_reg = reg
        best_lrate = lrate
        best_size = hidden_size

      print "Best accuracy so far is:", best_accuracy 
      print "With an average loss of:", best_loss

print "------------DONE-------------"
print "------------!!!!-------------"
print 'The best accuracy overall is', best_accuracy 
print "With an average loss of:", best_loss
print 'Regularization:', best_reg
print 'Learningrate:', best_lrate
print 'Hidden_Size:', best_size

plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()


# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc


#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################


# visualize the weights of the best network
show_net_weights(best_net)

# **We will give you extra bonus point for every 1% of accuracy above 52%.**

# 

test_acc = (best_net.predict(X_test) == y_test).mean()
print 'Test accuracy: ', test_acc


# Findings: Don´t change learning_decay_rate

# The best accuracy overall is 0.483
# With an average loss of: 1.45760862326
# Regularization: 0.03
# Learningrate: 0.001
# Hidden_Size: 100
# Test accuracy:  0.478
