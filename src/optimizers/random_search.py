import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *
from src.net_trainer import train_net

# get FER2013_data
fer2013_data = get_FER2013_data()

# defaut parameters
default_learning_rate = 5e-4
default_num_classes = 7
default_momentum = 0.5
default_hidden_units = 1
default_batch_size = 500
default_num_epochs = 1
default_lr_decay = 0.95
default_update_rule = "sgd_momentum"

# set number of iterations
num_iterations = 2

def optimize_learning_rate():

    # define item being optimized
    optim_param = "learning rate"
    
    # define filename template
    filename = "src/optimizers/outputs/random_search/learning_rate.txt"
    
    # define optim variable's type and range
    optim_var_type = np.float64
    low  = 5e-3
    high = 1e-6
    
    # append optimization info to file
    title = "Learning rate optimizer with 512 hidden units and momentum 0.5 \n"
    csv_format = "learning rate, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)

    # start optim
    for i in range(num_iterations):

        # randomly draw optim variable from the range low-high
        optim_variable = np.random.randint(low = low, high = high, dtype = optim_var_type)
        
        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = [default_hidden_units],
                           input_dim       = 48 * 48 * 3,
                           learning_rate   = optim_variable,
                           update_rule     = default_update_rule,
                           momentum        = default_momentum,
                           num_epochs      = default_num_epochs,
                           batch_size      = default_batch_size,
                           lr_decay        = default_lr_decay)
        
        # append results of iteration
        result = str.format('{0:6f}', optim_variable) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n"
        append_to_file(filename, result)
                    
    # generate plot of optimization
    plot_data(filename, title, "random_search")

def optimize_momentum():

    # define item being optimized
    optim_param = "momentum"
    
    # define filename template
    filename = "src/optimizers/outputs/random_search/momentum.txt"
    
    # define range on optim variable, and initiate it
    start  = 0.0
    end = 1.0
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Momentum optimizer with 512 hidden units and momentum 0.5 \n"
    csv_format = "momentum, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)

    # start optim
    for i in range(num_iterations):

        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = [default_hidden_units],
                           input_dim       = 48 * 48 * 3,
                           learning_rate   = default_learning_rate,
                           update_rule     = default_update_rule,
                           momentum        = optim_variable,
                           num_epochs      = default_num_epochs,
                           batch_size      = default_batch_size,
                           lr_decay        = default_lr_decay)
        
        # append results of iteration
        result = str.format('{0:6f}', optim_variable) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n"
        append_to_file(filename, result)
        
        # increment learning rate
        optim_variable += incr
            
    # generate plot of optimization
    plot_data(filename, title, "random_search")


def optimize_hidden_units():

    # define item being optimized
    optim_param = "hidden units"
    
    # define filename template
    filename = "src/optimizers/outputs/random_search/hidden_units.txt"
    
    # define range on optim variable, and initiate it
    start  = 50
    end = 2550
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Hidden units optimizer with 512 hidden units and momentum 0.5 \n"
    csv_format = "hidden units, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)

    # start optim
    for i in range(num_iterations):


        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = [int(optim_variable)],
                           input_dim       = 48 * 48 * 3,
                           learning_rate   = default_learning_rate,
                           update_rule     = default_update_rule,
                           momentum        = default_momentum,
                           num_epochs      = default_num_epochs,
                           batch_size      = default_batch_size,
                           lr_decay        = default_lr_decay)

        # append results of iteration
        result = str.format('{0:6f}', optim_variable) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n"
        append_to_file(filename, result)
        
        # increment learning rate
        optim_variable += incr
            
    # generate plot of optimization
    plot_data(filename, title, "random_search")


optimize_learning_rate()
#optimize_momentum()
#optimize_hidden_units()
