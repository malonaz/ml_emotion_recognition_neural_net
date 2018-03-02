import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver

def train_net(data,
              hidden_dims,
              input_dim,
              num_classes,
              dropout = 0,
              reg = 0,
              learning_rate = 5e-4,
              momentum = 0,
              update_rule_solver = 'sdg',
              plot = False,
              pickle = False):

    """ 
    Uses a solver instance to train a neural net on the given dataset.
    args:
    - data: dictionary with X_train, y_train, X_test, and y_test
    - plot: if True, generates a plot and saves in the given folder
    - pickle: if True, pickles the model in the given folder
    """

    # initialize net
    model = FullyConnectedNet(hidden_dims,
                              input_dim,
                              num_classes,
                              dropout,
                              reg)

    # initialize solver
    solver = Solver(model,
                    data,
                    update_rule  = update_rule_solver,
                    optim_config = {'learning_rate': learning_rate,
                                    'momentum': momentum},
                    lr_decay     = 0.95,
                    num_epochs   = 20,
                    batch_size   = 100,
                    print_every  = 100)
    

                              
        
                       




