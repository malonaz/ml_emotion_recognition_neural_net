import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data, load_data, save_net_info



def train_fer2013_net(hidden_units = 3500, learning_rate = 5e-4, momentum = 0, data = None, save_net = False):
    """
    Uses a Solver instance to train a neural net on the FER2013 dataset.
    Args:
    - data: FER2013 (X_train, y_train, X_test, y_test) dataset.
            If None, the procedure processes all the images which is computationally expensive.
    - plot: if True, generates a plot, saves it it in the the nets/fer2013 folder and shows it.
    - pickle: if True, pickles the model in the nets/fer2013 folder.

    Returns:
    - solver: the solver with the trained model
    """
    
    if data is None:
        # get FER2013 data
        data = get_FER2013_data()

    
    # intialize net
    model = FullyConnectedNet([hidden_units],
                              input_dim      = 48*48*3,
                              num_classes    = 7,
                              dropout        = 0,   # removed regularisation
                              reg            = 0.0, # removed regularisation
                              dtype          = np.float32)
        
        
    # initialize solver
    solver = Solver(model,data,
                    update_rule  = 'sgd_momentum',
                    optim_config = {'learning_rate': learning_rate,
                                     'momentum': momentum},
                    lr_decay     = 0.95,
                    num_epochs   = 20,
                    batch_size   = 100,
                    print_every  = 100)
    
    # train the net 
    solver.train()


    if save_net:
        # test the net
        model.test(data["X_test"], data["y_test"])

        # save net info
        save_net_info("nets/fer2013_net", solver)


    return solver



def optimize_learning_rate():

    # get FER2013 data
    fer2013_data = load_data()
    
    # initial learning_rate
    lr = 1e-4

    # open file and write what this will be about
    f = open("optimizers_output/l1noseed.txt", "a")
    f.write("Learning rate optimizer with 512 hidden units and momentum 0.5 \n")
    f.write("learning rate, best validation rate accuracy\n")
    f.close()
    
    while(True):
        

        solver = train_fer2013_net(hidden_units    = 512,
                                   data            = fer2013_data,
                                   learning_rate   = lr,
                                   momentum        = 0.5)
        # write result of iteration
        f = open("optimizers_output/l1noseed.txt", "a")
        f.write(str(lr) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n")
        f.close()
        
        # decrement learning rate
        lr -= .1e-4
            
    f.close()
            


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

#optimize_learning_rate()
#optimize_momentum()
#optimize_hidden_units()


def plot_data(param_optimized_name, filename):
    """
    generates a plot of the data at the filename
    """
    
    data = np.genfromtxt(filename, delimiter = ',', skip_header = 1, names= True)
    x,y = data.dtype.names

    fig = plt.figure()
    plt.plot(data[x], data[y])
    fig.suptitle("Title goes here")
    plt.xlabel(x)
    plt.ylabel(y)
    
    plt.savefig("optimizers_output/" + param_optimized_name + ".png", bbox_inches='tight')
    
    
plot_data("Optimizing the number of hidden units", "optimizers_output/h1noseed.txt")
plot_data("Optimizing the learning rate", "optimizers_output/l1noseed.txt")
plot_data("Optimizing the momentum", "optimizers_output/m1noseed.txt")
