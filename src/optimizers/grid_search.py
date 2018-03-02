import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *

# get FER2013_data
#data = get_FER2013_data()


def optimize_learning_rate():

    # define filename template
    filename = "src/optimizers/ouputs/grid_search/learning_rate.txt"
    
    # initial learning_rate
    lr = 1e-4

    # append optimization info to file
    title = "Learning rate optimizer with 512 hidden units and momentum 0.5 \n"
    csv_format = "learning rate, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)

    # start optim
    while(True):
        
        solver = train_fer2013_net(hidden_units    = 512,
                                   data            = fer2013_data,
                                   learning_rate   = lr,
                                   momentum        = 0.5)
        
        # append results of iteration
        result = str.format('{0:6f}',lr) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n"
        append_to_file(filename, result)
        
        # decrement learning rate
        lr -= .1e-4
            
    # generate plot of optimization
    plot_data("Optimizing the number of hidden units", "optimizers_output/h1noseed.txt")
            


def optimize_momentum():

    # get FER2013 data
    fer2013_data = load_data()
    
    # initial momentum 
    m = 0.0
    
    # open file and write what this will be about
    f = open("optimizers_output/m1noseed.txt", "a")
    f.write("Momentum optimizer with 512 hidden units and 5e-4 learning rate \n")
    f.write("momentum, best validation rate accuracy\n")
    f.close()
    
    while(m < 1.1):

        solver = train_fer2013_net(hidden_units    = 512,
                                   data            = fer2013_data,
                                   learning_rate   = 5e-4,
                                   momentum        = m)

        # write result of iteration
        f = open("optimizers_output/m1noseed.txt", "a")
        f.write(str(m) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n")
        f.close()
        
        # increment momentum
        m += .1


    f.close()
    plot_data("Optimizing the learning rate", "optimizers_output/l1noseed.txt")


    

def optimize_hidden_units():

    # get FER2013 data
    fer2013_data = load_data()
    
    # initial hidden unit numbers 
    hu = 50

    # open file and write what this will be about
    f = open("optimizers_output/h1noseed.txt", "a")
    f.write("Hidden units optimizer with 0.5 momentum and 5e-4 learning rate \n")
    f.write("hidden units, best validation rate accuracy\n")
    f.close()
    
    while(True):
        
        solver = train_fer2013_net(hidden_units = hu,
                                   data = fer2013_data,
                                   learning_rate = 5e-4,
                                   momentum = 0.5 )


        # write result of iteration
        f = open("optimizers_output/h1noseed.txt", "a")
        f.write(str(hu) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n")
        f.close()

        # increment hidden units
        hu += 50


    f.close()
    plot_data("Optimizing the momentum", "optimizers_output/m1noseed.txt")

#optimize_learning_rate()
#optimize_momentum()
#optimize_hidden_units()





