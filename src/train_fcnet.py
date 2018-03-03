import numpy as np
import matplotlib.pyplot as plt


from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *

import pickle


def train_cifar10_net(save_net = False):
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

    if save_net:
        
        # test the net and save its training, validation and testing accuracies.
        # get training accuracy
        train_acc = str.format("{0:.2f}", solver.check_accuracy(data["X_train"], data["y_train"]) * 100) + "\%"
        
        # get validation accuracy
        val_acc = str.format("{0:.2f}", solver.best_val_acc * 100) + "\%"
        
        # get testing accuracy
        test_acc = str.format("{0:.2f}", solver.check_accuracy(data["X_test"], data["y_test"]) * 100) + "\%"

        text = "Accuracies: " + train_acc + " training, " + val_acc + " validation  \& " + test_acc + " testing."

        # write to file
        append_to_file("nets/train_net/info.tex", text, mode =  "w")
        
        # save net info
        save_net_info("nets/train_net", solver)
    


train_cifar10_net(save_net = True)

