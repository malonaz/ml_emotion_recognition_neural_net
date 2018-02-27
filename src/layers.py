import numpy as np


def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    The input X has shape (N, d_1, ..., d_K) and contains N samples with each
    example X[i] has shape (d_1, ..., d_K). Each input is reshaped to a
    vector of dimension D = d_1 * ... * d_K and then transformed to an output
    vector of dimension M.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data
    """

    # get N and compute D = d_1 * ... * d_k
    N = X.shape[0]
    D = np.prod(X.shape[1:])
    
    # reshape X with dimensions: N x D
    reshapedX = X.reshape(N, D)

    # Compute WX. dimensions: (N x D) dot (D x M) = N x M
    XW = np.dot(reshapedX, W)

    # return biased XW
    return XW + b


def linear_backward(dout, X, W, b):
    """
    Computes the backward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d_1, ..., d_K), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A nump array of shape (M,), gradient with respect to b
    """

    # get N and compute D = d_1 * ... * d_k
    N = X.shape[0]
    D = np.prod(X.shape[1:])

    # for dW, derivate with respect to W_i is x_i

    # flatten X
    reshapedX = X.reshape(N, D)

    # d(sigma(W_all*x_all) + b)/dxi = W_i. must reshape dX as specs require it
    # (N, M) x (M x D) = N x D
#    print ("dout: ", dout.shape, " and W.T: ", W.transpose().shape)
    dX = np.dot(dout, W.transpose()).reshape(X.shape)
    
    # d(sigma(W_all*x_all)+ b )/dWi = x_i. Why don't we divide by number of examples?
    dW = np.dot(reshapedX.transpose(), dout)

    # d(sigma(W_all*x_all) + b)/db = 1. Collapse matrix vertically
    db = dout.sum(axis = 0)
         
    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """
    
    # Must use copy in numpy to avoid pass by reference.
    out = X.copy()

    # for all entries in out, x_ij = max(x_ij, 0)
    out[out < 0] = 0

    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    
    # Must use copy in numpy to avoid pass by reference.
    dX = dout.copy()
 
    # d(ReLU(x)/dx = 0 if x < 0 and 1 otherwise
    dX[X <= 0] = 0

    return dX


def dropout_forward(X, p=0.5, train=True, seed = None):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    
    # do no perform dropout if in test mode. Mask is None
    if not train:
        return X, None

    # seed random number generator
    if seed:
        np.random.seed(seed)
    
    # generate matrix of same size as X filled with values drawn from uniform distribution of [0, 1]
    mask = np.random.uniform(size = X.shape)

    # map the distribution to 1s and 0s as per the given probability
    mask[mask <= p] = 0
    mask[mask > p] = 1

    # now our expected output for each neuron is p*x. multiply through by 1/(1-p) to get expected output = x
    mask *= 1/(1 - p)

    # apply mask to out
    out = X * mask

    return out, mask


def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """

    
    # do not perform dropout if in test mode
    if not train:
        return dout

    # apply mask to dout
    dX = dout * mask
    
    return dX
