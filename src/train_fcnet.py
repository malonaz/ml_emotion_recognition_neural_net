import numpy as np
import matplotlib.pyplot as plt


from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

import pickle


def train_cifar10_net(save_net_info = False):
    """
    Uses a Solver instance to train a TwoLayerNet that achieves at least 50% 
    accuracy on the validation set.
    """

    # get CIFAR10 data
    data = get_CIFAR10_data()
        
    # intialize net
    model = FullyConnectedNet([100],
                              input_dim      = 32*32*3,
                              num_classes    = 10,
                              dropout        = 0,
                              reg            = 0.0)
        
        
    # initialize solver
    solver = Solver(model,data,
                    update_rule  = 'sgd',
                    optim_config = {'learning_rate': 1e-3},
                    lr_decay     = 0.95,
                    num_epochs   = 20,
                    batch_size   = 100,
                    print_every  = 100)
    
    # train the net 
    solver.train()


    if save_net_info:
        # test the net
        model.test(data["X_test"], data["y_test"])

        # save net info
        save_net_info("nets/train_net", solver)



train_cifar10_net(save_net_info = True)

