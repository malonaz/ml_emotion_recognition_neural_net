import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

# directory where the pickled batches are located
DIR = "datasets/cifar-10-batches-py/"

# contains the filenames of the 5 dataset batches
BATCHES_FILENAMES = [DIR + "data_batch_" + str(x + 1) for x in range(5)]

# contains the filename of the batches's meta file
BATCH_META_FILENAME = DIR + "batches.meta"

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


# unpicle batch meta info into a list
batch_meta = unpickle(BATCH_META_FILENAME)
examples_per_batch, class_names, num_vis = batch_meta.values() 

# unpickled batches into a list
batches = [unpickle(filename) for filename in BATCHES_FILENAMES]

# extract examples and data from batches, in tuples (examples, labels)
data = [(batch[list(batch.keys())[2]], batch[list(batch.keys())[1]]) for batch in batches]


overtfit_fcnet = FullyConnectedNet([10,10,10] , input_dim=32*32*3, num_classes=10,
                                   dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                                   seed = 1)

