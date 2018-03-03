import numpy as np


from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *

import pickle


def train_overfit_net(save_net = False):


    # get CIFAR10 data
    data = get_CIFAR10_data()
        
    # subsample the data. 50 draws
    indices_to_select = np.random.randint(0, len(data["X_train"]), 50)

    # extract the samples
    data["X_train"] = data["X_train"][indices_to_select]
    data["y_train"] = data["y_train"][indices_to_select]
    
    # intialize net
    model = FullyConnectedNet([50],
                              input_dim      = 32*32*3,
                              num_classes    = 10,
                              dropout        = 0,
                              reg            = 0.0,
                              weight_scale   = 1e-2)
        
        
    # initialize solver
    solver = Solver(model,data,
                    update_rule  = 'sgd',
                    optim_config = {'learning_rate': 5e-4},
                    lr_decay     = 0.85,
                    num_epochs   = 20,
                    batch_size   = 5,
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
        append_to_file("nets/overfit_net/info.tex", text, mode =  "w")
        
        # save net info
        save_net_info("nets/overfit_net", solver)
        

train_overfit_net(save_net = True)


