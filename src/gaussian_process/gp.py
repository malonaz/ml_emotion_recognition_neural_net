import numpy as np
import matplotlib.pyplot as plt
import GPy
import GPyOpt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data, load_data



# get FER2013 data
data = get_FER2013_data()


def train_net(hyperparameters):

    print(hyperparameters)
    # extract hyperparameters
    lr, m, d, hu = hyperparameters[0]
    lr = np.exp(lr) # log scale
    hu = int(hu)
    
    # intialize net
    model = FullyConnectedNet([hu],
                              input_dim      = 48*48*3,
                              num_classes    = 7,
                              dropout        = d,   # removed regularisation
                              reg            = 0.0, # removed regularisation
                              dtype          = np.float32)
        
        
    # initialize solver
    solver = Solver(model, data,
                    update_rule  = 'sgd_momentum',
                    optim_config = {'learning_rate': lr,
                                     'momentum': m},
                    lr_decay     = 0.95,
                    num_epochs   = 20,
                    batch_size   = 100,
                    print_every  = 100)

    
    # train the net 
    solver.train()
    
    return solver.best_val_acc


opt_iter = 100
domain = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (-11, -6)},
          {'name': 'momentum',  'type': 'continuous', 'domain': (0, 1)},
          {'name': 'dropout',  'type': 'continuous', 'domain': (0, 1)},
          {'name': 'hidden_units',  'type': 'continuous', 'domain': (50, 1000)}]



n_dim = len(domain)
n_initial = 5 * n_dim
bayes_opt = GPyOpt.methods.BayesianOptimization(f = train_net,
                                                domain = domain,
                                                model_type = 'GP',
                                                acquisition_type = 'EI',
                                                exact_feval = True,
                                                initial_design_numdata = n_initial,
                                                initial_design_type = 'random',
                                                num_cores = 2)  

bayes_opt.run_optimization(max_iter = opt_iter)
bayes_opt.plot_convergence()

min_index = np.argmin(bayes_opt.Y)
x_best = bayes_opt.X[min_index,:]
y_best = bayes_opt.Y[min_index,:]

print(x_best)
print(y_best)
