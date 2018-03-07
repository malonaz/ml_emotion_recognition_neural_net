import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import *
from src.net_trainer import train_net

# get FER2013_data
fer2013_data = load_data()

# defaut parameters
default_learning_rate = 0.001
default_num_classes = 7
default_momentum = 0.95
default_hidden_units = 128
default_batch_size = 128
default_num_epochs = 25
default_lr_decay = 0.95
default_update_rule = "sgd_momentum"

# set number of iterations
num_iterations = 1


def optimize_learning_rate():

    # define item being optimized
    optim_param = "learning rate"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/learning_rate.txt"
    
    # define range on optim variable, and initiate it
    start  = 1e-4
    end = 1e-7
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
                           hidden_dims     = [316, 100],
                           input_dim       = 48 * 48 * 3,
                           learning_rate   = default_learning_rate,
                           update_rule     = default_update_rule,
                           momentum        = default_momentum,
                           num_epochs      = default_num_epochs,
                           batch_size      = default_batch_size,
                           lr_decay        = default_lr_decay)


        # append results of iteration
        result = str.format('{0:6f}', optim_variable) + ", " + str.format('{0:6f}', solver.best_val_acc) + "\n"
        print (result)
        #append_to_file(filename, result)
        
        # increment learning rate
        optim_variable += incr


def get_optimal_learning_rate_info():

    # set data
    data = fer2013_data
    
    # optimal learning rate
    optimal_learning_rate = 6.5e-5
    
    solver = train_net(data            = data,
                       num_classes     = default_num_classes,
                       hidden_dims     = [316, 100],
                       input_dim       = 48 * 48 * 3,
                       learning_rate   = optimal_learning_rate,
                       update_rule     = default_update_rule,
                       momentum        = default_momentum,
                       num_epochs      = default_num_epochs,
                       batch_size      = default_batch_size,
                       lr_decay        = default_lr_decay)
    
        
    # test the net and save its training, validation and testing classification errors.
    
    # get training
    train_err = str.format("{0:.2f}", (1 - solver.check_accuracy(data["X_train"], data["y_train"])) * 100) + "\%"
    
    # get validation 
    val_err = str.format("{0:.2f}", (1 - solver.best_val_acc) * 100) + "\%"
    
    # get testing 
    test_err = str.format("{0:.2f}", (1 - solver.check_accuracy(data["X_test"], data["y_test"])) * 100) + "\%"
    
    text = "Classification error rates: " + train_err + " training, " + val_err + " validation  \& " + test_err + " testing."
    
    # write to file
    append_to_file("nets/optimal_learning_rate/info.tex", text, mode =  "w")
    
    # save net info
    save_net_info("nets/optimal_learning_rate", solver)



def get_optimal_net_info():

    # set data
    data = fer2013_data

    # optimal learning rate
    optimal_learning_rate = default_learning_rate
    
    solver = train_net(data            = data,
                       num_classes     = default_num_classes,
                       hidden_dims     = [316, 100],
                       input_dim       = 48 * 48 * 3,
                       learning_rate   = optimal_learning_rate,
                       update_rule     = default_update_rule,
                       momentum        = default_momentum,
                       num_epochs      = default_num_epochs,
                       batch_size      = default_batch_size,
                       lr_decay        = default_lr_decay)
    
        
    # test the net and save its training, validation and testing classification errors.
    
    # get training
    train_err = str.format("{0:.2f}", (1 - solver.check_accuracy(data["X_train"], data["y_train"])) * 100) + "\%"
    
    # get validation 
    val_err = str.format("{0:.2f}", (1 - solver.best_val_acc) * 100) + "\%"
    
    # get testing 
    test_err = str.format("{0:.2f}", (1 - solver.check_accuracy(data["X_test"], data["y_test"])) * 100) + "\%"
    
    text = "Classification error rates: " + train_err + " training, " + val_err + " validation  \& " + test_err + " testing."
    
    # write to file
    append_to_file("nets/optimal_net/info.tex", text, mode =  "w")
    
    # save net info
    save_net_info("nets/optimal_net", solver)

    ### now save confusion matrix on test data

    # get test data
    X_test, y_test = data["X_test"], data["y_test"]

    # get the prediction vector
    predictions = solver.model.loss(X_test).argmax(axis = 1)

    # get confusion matrix and save it
    confusion_matrix = get_confusion_matrix(predictions, y_test)

    np.savetxt("nets/optimal_net/confusion_matrix.tex", confusion_matrix, delimiter = ' & ', fmt = '%i', newline =' \\\\\n')

    # get recall and precision rates
    recall_rates, precision_rates = get_recall_precision_rates(confusion_matrix)
    
    # get F-1 score
    f_measures = get_f_measures(recall_rates, precision_rates)

    # put recall, precision and F-1 score together and save the composite matrix
    metrics = np.empty((3, recall_rates.shape[0]))
    metrics[0], metrics[1], metrics[2] = recall_rates, precision_rates, f_measures
    np.savetxt("nets/optimal_net/metrics.tex", metrics, delimiter = ' &', fmt = '%1.3f', newline = ' \\\\\n')
    

def optimize_dropout():

    # define item being optimized
    optim_param = "dropout rate"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/dropout_rate.txt"
    
    # define range on optim variable, and initiate it
    start  = 0.0
    end = 1.0
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "Dropout rate optimizer with 256 hidden units and momentum 0.5 \n"
    csv_format = "dropout rate, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)
    
    # start optim
    for i in range(num_iterations):
        
        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = [default_hidden_units],
                           input_dim       = 48 * 48 * 3,
                           dropout         = optim_variable,
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


        
def optimize_L2():

    # define item being optimized
    optim_param = "L2 regularization rate"
    
    # define filename template
    filename = "src/optimizers/outputs/grid_search/L2_rate.txt"
    
    # define range on optim variable, and initiate it
    start  = 0.0
    end = 1.0
    incr = (end - start)/num_iterations
    optim_variable = start
    
    # append optimization info to file
    title = "L2 regularization rate optimizer with 256 hidden units and momentum 0.9 \n"
    csv_format = "L2 rate, best validation rate accuracy\n"
    append_to_file(filename, title + csv_format)
    
    # start optim
    for i in range(num_iterations):
        
        # create and train a net
        solver = train_net(data            = fer2013_data,
                           num_classes     = default_num_classes,
                           hidden_dims     = [default_hidden_units],
                           input_dim       = 48 * 48 * 3,
                           reg             = optim_variable,
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

        


def generate_plots():
    # learning rate
    #filename = "src/optimizers/outputs/grid_search/learning_rate.txt"
    #title = "Learning rate optimizer with 256 hidden units and 0.9 momentum \n"
    #plot_data(filename, title, "grid_search")

    # dropout
    filename = "src/optimizers/outputs/grid_search/dropout_rate.txt"
    title = "dropout rate optimizer with 5e-4 learning rate and 0.9 momentum \n"    
    plot_data(filename, title, "grid_search")
    

    # L2
    #filename = "src/optimizers/outputs/grid_search/L2_rate.txt"
    #title = "L2 regularization optimizer with 5e-4 learning rate and 0.9 momentum \n"    
    #plot_data(filename, title, "grid_search")
    
    

    
get_optimal_net_info()
#optimize_learning_rate()
#generate_plots()
#optimize_dropout()
