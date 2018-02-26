import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""


def unpickle(filename):
    """
    args:
    - filename: the filename of the object to unpickle, relative to the main folder
    
    return:
    - obj: the object that was unpickled
    """
    # need for unpickling
    import pickle

    # open file name in binary mode
    f = open(filename, 'rb')

    # unpickle the object contained the file
    obj = pickle.load(f, encoding = 'bytes')

    return obj




# load the 5 batches
batch1 = unpickle("datasets/cifar-10-batches-py/data_batch_1")
overtfit_fcnet = FullyConnectedNet([10,10,10] , input_dim=32*32*3, num_classes=10,
                                   dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                                   seed = 1)

