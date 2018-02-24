import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    
    # compute K = -max(Wx_i), which will be used for numerical stability.
    # here since we are dealing with logs, we will compute logK directly for each row.
    logKs = logits.max(axis = 1).reshape(logits.shape[0], 1)
    
    # add the logKs to the appropriate rows to stabilize the logits
    # for each entry, raise the number e to the power of this entry to get the score
    stableScores = np.exp(logits + logKs)    

    # collapse matrix horizontally to get the sum of the scores of each sample
    sampleScoreSums = stableScores.sum(axis = 1).reshape(logits.shape[0], 1)

    # for each example, divide each class score by the sum of the scores of all classes
    # this turns the scores into a probability distribution over the classes: P(y_i = class_j|logits)
    classDistribution = stableScores / sampleScoreSums

    # compute likelihood: P(y_i = correctClass | logits)
    correctClassProbabilities = classDistribution[np.arange(classDistribution.shape[0]), y]

    # compute the negative log-likelihood 
    negLogLikelihood = -np.log(correctClassProbabilities)

    # loss = sigma(negloglikelihood)/N
    loss = negLogLikelihood.sum()/negLogLikelihood.shape[0]
    
    dlogits = None
    
    return loss, dlogits
