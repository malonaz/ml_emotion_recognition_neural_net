import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data, load_data



# get FER2013 data
data = get_FER2013_data()

opt_iter = 100
domain = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (-11, -6)},
          {'name': 'momentum',  'type': 'continuous', 'domain': (0, 1)},
          {'name': 'dropout',  'type': 'continuous', 'domain': (0, 1)},
          {'name': 'hidden_units',  'type': 'discrete', 'domain': (50, 1000)}]

