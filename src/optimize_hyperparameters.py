import numpy as np
import matplotlib.pyplot as plt


from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data

import pickle

def pickle_data():

    # get FER2013 data
    fer2013_data = get_FER2013_data()

    # dump the pickle
    pickle.dump(fer2013_data, open("datasets/FER2013/data.p", "wb"))

    
def load_data():
    return pickle.load(open("datasets/FER2013/data.p", "rb"))



def train_fer2013_net(learning_rate = 5e-4, momentum = 0, data = None, plot = False):
    """
    Uses a Solver instance to train a neural net on the FER2013 dataset
    """
    if data is None:
        # get FER2013 data
        data = get_FER2013_data()

        
    # intialize net
    model = FullyConnectedNet([500, 500],
                              input_dim      = 48*48*3,
                              num_classes    = 6,
                              dropout        = 0,   # removed regularisation
                              reg            = 0.0, # removed regularisation
                              dtype          = np.float32,
                              seed           = 1)
        
        
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
        
    # pickle net
    #pickle.dump(model, open("nets/fer2013_net/pickled_net.p", "wb"))

        
    if plot:

        plt.subplot(2, 1, 1)

        plt.subplot(2, 1, 1)
    
        plt.title("trainingloss")
        plt.plot(solver.loss_history, "o")
        plt.xlabel('Iteration')
        
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(solver.train_acc_history,'-o', label = 'train')
        plt.plot(solver.val_acc_history,'-o', label = 'val')
        plt.plot([0.5] * len(solver.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc = 'lower right')
        plt.gcf().set_size_inches(15, 12)

        # save figure
        plt.savefig("nets/fer2013_net/diagrams.png", bbox_inches='tight')

        # show figure
        plt.show()

        
    return solver


def optimize_learning_rate():

        # get FER2013 data
    fer2013_data = load_data()
    
    # initial learning_rate
    lr = 1e-4
    
    while(True):
        
        lr += .1e-4
        
        solver = train_fer2013_net(data = fer2013_data, learning_rate = lr)
        
        with open("output_lr2.txt", "a") as f:
            f.write("learning rate: " + str(lr) + " val_acc: " + str(solver.best_val_acc) + "\n")
            
            
            

def optimize_momentum():

    # get FER2013 data
    fer2013_data = load_data()
    
    # initial learning_rate
    m = 0.1

    while(m < 1.1):

        m += .1
        
        solver = train_fer2013_net(data = fer2013_data, learning_rate = 5e-3, momentum = m )

        with open("output_m3.txt", "a") as f:
            f.write("momentum: " + str(m) + " val_acc: " + str(solver.best_val_acc) + "\n")




    
optimize_momentum()