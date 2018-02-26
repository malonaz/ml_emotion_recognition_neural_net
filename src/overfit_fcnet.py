import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

# get CIFAR10 data
""" keys of data
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
"""
data = get_CIFAR10_data()


# intialize net
model = FullyConnectedNet([10,10,10] , input_dim=32*32*3, num_classes=10,
                          dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                          seed = 1)
solver = Solver(model, data)
