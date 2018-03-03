import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *
from src.net_trainer import train_net

# get FER2013_data
fer2013_data = load_data()

# defaut parameters
default_learning_rate = 5e-4
default_num_classes = 7
default_momentum = 0.5
default_hidden_units = 512
default_batch_size = 500
default_num_epochs = 20
default_lr_decay = 0.95
default_update_rule = "sgd_momentum"

# set ranges of optim params
range_learning_rate = (1e-7, 1e-4)

range_momentum = (0.0, 1.0)

range_batch_size = (20, 200)

range_num_epochs = (20, 200)

range_lr_decay = (0.5, 1.0)

range_hidden_units = (100, 500)

range_num_hidden_layers = (1, 4)


# set number of iterations
num_iterations = 200

def random_search():

    print ("best_val_accuracy, learning_rate, momentum, lr_decay, num_epochs, batch_size, \
num_hidden_layers, hidden_dim1, hidden_dim2, hidden_dim3\n")

    # used to hold the best val and its parameters
    best_val_acc = 0.0
    best_params = "none"

    # start optim
    for i in range(num_iterations):

        # randomly draw optim variables from the range low-high
        # floats
        learning_rate = np.random.uniform(low = range_learning_rate[0], high = range_learning_rate[1])
        momentum = np.random.uniform(low = range_momentum[0], high = range_momentum[1])
        lr_decay = np.random.uniform(low = range_lr_decay[0], high = range_lr_decay[1])
        # integers
        num_epochs = np.random.randint(low = range_num_epochs[0], high = range_num_epochs[1])
        batch_size  = np.random.randint(low = range_batch_size[0], high = range_batch_size[1])
        num_hidden_layers = np.random.randint(low = range_num_hidden_layers[0], high = range_num_hidden_layers[1])
        hidden_dims = [np.random.randint(low = range_hidden_units[0], high = range_hidden_units[1])
                       for i in range(num_hidden_layers)]
        
        
        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = hidden_dims,
                           input_dim       = 48 * 48 * 3,
                           learning_rate   = learning_rate,
                           update_rule     = default_update_rule,
                           momentum        = momentum,
                           num_epochs      = num_epochs,
                           batch_size      = batch_size,
                           lr_decay        = lr_decay,
                           verbose         = False)
        
        # append results of iteration
        results = str.format('{0:6f}', solver.best_val_acc) + ', ' +\
                  str.format('{0:6f}', learning_rate) + ', ' + \
                  str.format('{0:6f}', momentum) + ', ' + \
                  str.format('{0:6f}', lr_decay) + ', ' +\
                  str(num_epochs) + ', ' +\
                  str(batch_size) + ', ' +\
                  str(num_hidden_layers)

        for i in range(3):
            if i < num_hidden_layers:
                results += ', ' + str(hidden_dims[i])
            else:
                results += ', -1'
        
                  
        results += "\n"
        print (results)

        # update best_params
        if solver.best_val_acc > best_val_acc:
            best_params = results
            best_val_acc = solver.best_val_acc
        
        
    print (best_params)

random_search()
