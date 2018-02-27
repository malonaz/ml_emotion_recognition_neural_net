import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        # assign a new dict to config
        config = {}

    # set learning rate if none exist in the dictionary
    config.setdefault('learning_rate', 1e-2)

    # update the weights. We perform in-place updates for efficiency
    w -= config['learning_rate'] * dw
    
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        # assign a new dict to config
        config = {}

    # set default rate and momentum if they do not exist in the dictionary
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)

    # get momemtum and rate
    lr = config.get('learning_rate', 1e-2)
    momentum = config.get('momentum', 0.9)

    # get the dict's velocity matrix or generate one if none exist, filled with zeroes
    velocity = config.get('velocity', np.zeros_like(w))

    # update velocity
    velocity = momentum * v - learning_rate * dw

    # store the velocity again
    config['velocity'] = velocity

    # update weights
    next_w = w + velocity

    return next_w, config
