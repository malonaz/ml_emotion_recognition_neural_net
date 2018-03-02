from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread 
#from imread import imread
import platform
import matplotlib.pyplot as plt

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = "datasets/cifar-10-batches-py/data_batch_" + str(b)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch("datasets/cifar-10-batches-py/test_batch")
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):

    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

    # get raw data
    raw_data = load_CIFAR10(cifar10_dir)

    # return processed data
    return process_data(raw_data, num_training, num_validation, num_test, subtract_mean)

    
def process_data(raw_data, num_training = None, num_validation = None, num_test = None,
                     subtract_mean = True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    X_train, y_train, X_test, y_test = raw_data

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def get_image(filename):
    """
    opens image using imread and returns a numpy array contains the image
    """
    return imread(filename)


def get_FER2013_data():

    # place labels_public's lines into a list
    with open("datasets/FER2013/labels_public.txt", "r") as labels_file:
        # remove first line because it is just info
        filenames_and_labels = labels_file.readlines()[1:]

    # how many examples total
    num_examples = len(filenames_and_labels)
    
    # find out how many training example. 
    num_training = 0
    for i in range(num_examples):
        if filenames_and_labels[i][0 + 1] == "e":
            num_training = i
            break

    # compute num_test
    num_test = num_examples - num_training


    # split filenames and labels
    filenames, labels = zip(*map(lambda x: x.split(','), filenames_and_labels))

    # free memory
    del filenames_and_labels

    # process labels
    y_train  = np.array(labels[:num_training]).reshape((num_training, ))
    y_test   = np.array(labels[num_training:]).reshape((num_test, ))

    # process filenames
    X_train, X_test = process_images(filenames, num_training, num_test)
    
                
    # package into one list, after casting the y matrices to integers
    raw_data = [X_train, y_train.astype(int, copy = False), X_test, y_test.astype(int, copy = False)]

    return process_data(raw_data,
                        num_training = num_training - 1200,
                        num_validation = 1200,
                        num_test = len(y_test))


def process_images(filenames, num_training, num_test = 0):
    """
    Processes all the images with the filenames.
    
    Args:
    - filenames: list of filenames relative to main folder, with training filenames
                 first and test filenames second if any.
    - num_training: the number of given filenames that are training examples
    - num_testing:  the number of given filenames that are testing examples
    
    Returns:
    - X_train: matrice containing the training examples as matrices
    - X_test: matrice containing the testing examples as matrices if any
    """
    
    if num_test == 0:
        num_training = len(filenames)
        
    # now find out the size of an element using the first image element
    filename = "datasets/FER2013/" + filenames[0]
    image_matrix_shape = list(get_image(filename).shape)
    
    # used to contain the appropriate data. initialized to empty arrays
    X_train  = np.empty(([num_training] + image_matrix_shape))
    X_test   = np.empty(([num_test] + image_matrix_shape))

    
    # iterate through each element of the list
    for i in range(num_training + num_test):

        # get image matrix
        image_matrix = get_image("datasets/FER2013/" + filenames[i])

        # append the matrix and the label to the appropriate array
        if i < num_training:
            X_train[i] = image_matrix

        else:
            X_test[i - num_training] = image_matrix

    return X_train, X_test



def load_data():
    """
    Unpickles pickled data from the FER2013 dataset and
    returns it as (X_train, y_train, X_test, y_test)
    """
    
    return pickle.load(open("datasets/FER2013/data.p", "rb"))


def pickle_data():
    """
    Extracts the data from the FER2013 dataset and pickles it
    """

    # get FER2013 data
    fer2013_data = get_FER2013_data()

    # dump the pickle
    pickle.dump(fer2013_data, open("datasets/FER2013/data.p", "wb"))


def save_net_info(folder, solver):
    """
    Saves a matplot diagram of training losses, training and validation accuracy.
    Saves a pickled version of the model.
    Saves the testing accuracy.
    """

    # get model
    model = solver.model

    # pickle model
    pickle.dump(model, open(folder + "/pickled_net.p", "wb"))

    # create and save plot
    plt.subplot(2, 1, 1)
    
    plt.subplot(2, 1, 1)
    
    plt.title("Training Loss")
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
    
    plt.savefig("nets/overfit_net/diagrams.png", bbox_inches='tight')


    # save accuracy info if any
    if model.test_acc is not None:
        f  = open(folder + "/info.tex", "w")
        f.write("Validation set accuracy: " + str.format("{0:.2f}", solver.best_val_acc * 100) + "\%")
        f.write(" \& testing set accuracy: " + str.format("{0:.2f}", model.test_acc * 100) + "\%")
        f.close()

def append_to_file(filename, text):
    """
    Appends text to given file
    """
    
    f = open(filename, "a")
    f.write(text)
    f.close()
    

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
