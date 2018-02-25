import numpy as np
from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    
    # initialize weights from a normal distribution with sd equal to weight_scale
    W = np.random.normal(scale = weight_scale, size = (n_in, n_out))

    # initialize biases to 0. One for each next layer neuron
    b = np.zeros((n_out,))
    
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the random
        initialisation (see manual).
        """

        # create a layers dimension list
        layer_dims = hidden_dims + [num_classes]
        
        # used to store the number of input nodes into each output of the next layer
        # initialized to the input dim for the first layer
        n_in = input_dim

        # iterate 
        for i in range(len(layer_dims)):

            # get the number of output for each input
            n_out = layer_dims[i]

            # print ("layer" + str(i + 1) + "n_in: ", n_in, " and n_out: ", n_out)

            # initialize weights and biases
            W, b = random_init(n_in, n_out, weight_scale, dtype)

            # store weight and b into params using appropriate name
            self.params["W" + str(i + 1)] = W
            self.params["b" + str(i + 1)] = b

            # update number of input node into each output of the next layer
            n_in = n_out                
            
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = {"train": self.use_dropout, "p": dropout, "seed": seed}
                
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        
        # get num of hidden layers (do not count the output layer)
        num_hidden_layers = self.num_layers - 1

        # store the first layer's input in the linear cache
        linear_cache["X1"] = X

        # iterate through all hidden layers, starting with index 1
        for i in range(num_hidden_layers):

            # compute this layer's X W and b names
            X, W, b = "X" + str(i + 1), "W" + str(i + 1), "b" + str(i + 1)

            # compute next layer's X W and b names
            X_next, W_next, b_next = "X" + str(i + 2), "W" + str(i + 2), "b" + str(i + 2)
                        
            # perform linear pass. output has dimensions M x N
            linear_out = linear_forward(linear_cache[X], self.params[W], self.params[b])
            
                        
            # perform relu -> dropout
            if i < num_hidden_layers - 1:
                # perform ReLU
                relu_out = relu_forward(linear_out)

                # perform dropout
                out, mask = dropout_forward(relu_out, \
                                            self.dropout_params["p"], \
                                            self.dropout_params["train"], \
                                            self.dropout_params["seed"])
                
                # add this layer's output as input to the next layer in the linear cache
                linear_cache[X_next] = out

                
            # this final iteration's output are the scores
            else:
                scores = out
                
            
        # if y is None then we are in test mode so just return scores
        if y is None:
            return scores


        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
    
        # [linear - relu - (dropout)] x (N - 1) - linear - softmax

        
        # perform softmax. should I log scores?
        loss, dlogits = softmax(scores, y)
        
        # used to store the input of the next layer to be processed
        dout = dlogits
        
        print (scores.shape)

        
        #for key in self.params:
        #    print (key, self.params[key].shape)
        #for key in linear_cache:
        #    print (key, linear_cache[key].shape)
        
        
        for i in range(num_hidden_layers, 0, -1):

            # compute this layer's X, W and b names
            X, W, b = "X" + str(i), "W" + str(i), "b" + str(i)
            
            # perform linear backward
            dX, dW, db = linear_backward(dlogits, linear_cache[X], self.params[W], self.params[b])
            
                                         
        # lets compute gradients
        grads = dict()
        

        # perform linear backward
        
        return loss, grads
