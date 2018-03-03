import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *
from src.net_trainer import train_net

# get FER2013_data
fer2013_data = load_data()

# defaut parameters
default_learning_rate = 1e-3
default_num_classes = 7
default_momentum = 0.9
default_hidden_units = 256
default_batch_size = 60
default_num_epochs = 20
default_lr_decay = 0.90
default_update_rule = "sgd_momentum"

# set number of iterations
num_iterations = 20

def optimize_learning_rate():

    # define item being optimized
    optim_param = "learning rate"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/learning_rate.txt"
    
    # define range on optim variable, and initiate it
    start  = 1e-3
    end = 1e-6
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Learning rate optimizer with 256 hidden units and momentum 0.5 \n"
    csv_format = "learning rate, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)
    
    # start optim
    for i in range(num_iterations):
        
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
        
        # increment learning rate
        optim_variable += incr


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


        
def optimize_momentum():

    # define item being optimized
    optim_param = "momentum"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/momentum.txt"
    
    # define range on optim variable, and initiate it
    start  = 0.0
    end = 1.0
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Momentum optimizer with 256 hidden units and momentum 0.5 \n"
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
    plot_data(filename, title, "grid_search")


def optimize_hidden_units():

    # define item being optimized
    optim_param = "hidden units"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/hidden_units.txt"
    
    # define range on optim variable, and initiate it
    start  = 50
    end = 2550
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Hidden units optimizer with 256 hidden units and momentum 0.5 \n"
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
    plot_data(filename, title, "grid_search")


def generate_plots():
    # learning rate
    filename = "src/optimizers/outputs/grid_search/learning_rate.txt"
    title = "Learning rate optimizer with 256 hidden units and 0.5 momentum \n"
    plot_data(filename, title, "grid_search")

    # momentum
    filename = "src/optimizers/outputs/grid_search/momentum.txt"
    title = "Momentum optimizer with 5e-4 learning rate and 0.5 momentum \n"    
    plot_data(filename, title, "grid_search")
    
    # hidden units
    title = "Hidden units optimizer with 5e-4 learning rate and 0.5 momentum \n"
    filename = "src/optimizers/outputs/grid_search/hidden_units.txt"
    plot_data(filename, title, "grid_search")


def optimize():
    optimize_learning_rate()
    optimize_momentum()
    optimize_hidden_units()


#generate_plots()    
