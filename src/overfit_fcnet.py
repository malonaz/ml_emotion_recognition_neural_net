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
model = FullyConnectedNet([200, 150] , input_dim=32*32*3, num_classes=10,
                          dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                          seed = 1)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 5e-2,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=100,
                print_every=100)
x = data["X_test"]


solver.train()

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
plt.legend(loc = 'lowerright')
plt.gcf().set_size_inches(15, 12)
plt.show()

