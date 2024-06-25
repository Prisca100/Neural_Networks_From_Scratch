import numpy as np


# Activations
def sigmoid(Z):
    """returns sigmoid activation of input vector x"""
    A = 1/(np.exp(-Z) + 1)
    return A
    
def sigmoid_backward(dA, cache):
    """returns computed gradient dZ"""
    Z = cache
    s = sigmoid(Z)

    # s(1-s) =  sigmoid derivative
    dZ = dA * s*(1-s)

    assert(dZ.shape == Z.shape)
    return dZ

def tanH():
    pass

def relU(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A

def reLU_backward(dA, cache):
    Z =  cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0

    return dZ

def softMax(Z):
    Z_max = np.max(Z)
    exp_z = np.exp(Z-Z_max)
    A = exp_z / np.sum(exp_z)
    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def softMax_backward(dA, cache):
    Z = cache
    s, _ = softMax(Z)
    s_diag = np.diag(s)
    s_outer = np.outer(s, s)
    jacobian = s_diag-s_outer
    dZ = dA * jacobian

    assert(dZ.shape == Z.shape)

    return dZ

    

x = np.array([1,2,3])
# print(relU(x))


# Image preprocessing
def image2Vector(image):
    if not isinstance(image,np.ndarray):
        image = np.array(image)
    print(image.shape)
    length, height, depth = image.shape
    if depth:
        return image.reshape(length*height*depth, 1)
    return image.reshape(length*height,1)


image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])


# print(f"img2vct = {str(image2Vector(image))}")



def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess
