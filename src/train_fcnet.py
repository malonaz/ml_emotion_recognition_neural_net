import numpy as np
import matplotlib.pyplot as plt


from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

import pickle


def train_cifar10_net(plot = False):
    """
    Uses a Solver instance to train a TwoLayerNet that achieves at least 50% 
    accuracy on the validation set.
    """

    # get CIFAR10 data
    data = get_CIFAR10_data()

    print (data.keys())
        
    # intialize net
    model = FullyConnectedNet([100],
                              input_dim      = 32*32*3,
                              num_classes    = 10,
                              dropout        = 0,
                              reg            = 0.0,
                              dtype          = np.float32,
                              seed           = 1)
        
        
    # initialize solver
    solver = Solver(model,data,
                    update_rule  = 'sgd',
                    optim_config = {'learning_rate': 5e-4},
                    lr_decay     = 0.95,
                    num_epochs   = 20,
                    batch_size   = 100,
                    print_every  = 100)
    
    # train the net 
    solver.train()
        
    # pickle net
    pickle.dump(model, open("nets/train_net/pickled_net.p", "wb"))

        
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
        plt.show()


train_cifar10_net(plot = True)

