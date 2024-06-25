import numpy as np
# loss functions
def L1(y_true, y_hat):
    """returns the sum of the absolute of the difference between predictions and actual values
        y_hat: predictions
        y_true: actual values
    """
    return sum(abs(y_true-y_hat))

def L2(y_true, y_hat):
    """ 
    Computes the vertical distance between the actual points and the fitted function
    returns the sum of the distances
    """
    x = y_true - y_hat
    # return np.dot(x,x)
    return sum(np.square(y_true-y_hat))
   
def cross_entropy(y_true, y_hat):
    """ y_true: array of actual values of dimension (1, number of examples)
        y_hat: array of predicted values. Same size as above
        returns cross_entropy: average of the log probabilities of y_hat
    """
    m = y_true.shape[1]
    logProps = np.dot(y_true, np.log(y_hat).T) + np.dot((1-y_true), np.log(1-y_hat).T)
    cost = np.squeeze((-1/m)*logProps)

    # assert(isinstance(cost, float))
    assert(cost.shape == ())
    return cost

def mean_square_error(y_true, y_hat):
    cost = np.square(y_true-y_hat)
    return np.sum(cost)
     
