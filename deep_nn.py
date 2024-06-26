import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid, tanH, softMax, relU, reLU_backward, sigmoid_backward, softMax_backward
from cost_functions import cross_entropy
from testCase_v4a import L_model_forward_test_case


def initialize_parameters(layer_dims):
    L = len(layer_dims)
    np.random.seed(3)
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.1
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A_prev, W, b):
    Z  = np.dot(W, A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    # cache = (A_prev, W, b)
    return Z

def linear_activation_forward(A_prev, W, b, activation):
    """
        Activation cache: Z 
        lin_cache: A_prev, W, b
    """
    Z = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relU(Z)
    # elif activation == "tanh":
    #     A, activation_cache = tanH(Z)
    # elif activation == "softmax":
    #     A, activation_cache = softMax(Z)
    else:
        print("\nActivation " +str(activation) + " not present. Available activations:\n1: \
              'softmax'\n2: 'sigmoid'\n3: 'relu'\n4: 'tanh'")
    
    lin_cache = (A_prev, W, b)
    cache = (lin_cache, Z)
    return A, cache

def forward_pass(X, parameters, hidden_activation="relu", output_activation="sigmoid"):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], hidden_activation)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], output_activation)
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches



def compute_cost(AL, Y, cost_func="cross_entropy"):
    m = Y.shape[1]
    if cost_func == 'cross_entropy':
        return cross_entropy(Y, AL)
    else:
        pass
 
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    # A_prev, W, b = linear_cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = reLU_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softMax_backward(dA, activation_cache)
    # dZ_prev = np.dot(W.T, dZ)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_pass(AL, Y, caches):
    "returns gradients of all W's and b's"
    grads = {}
 
    L = len(caches) #3
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    last_layer_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL,last_layer_cache, "sigmoid")
    # L = 3
    for l in reversed(range(L-1)):
        # 1
        current_cache = caches[l]
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)]  = linear_activation_backward(grads["dA"+str(l+1)], current_cache, 'relu')

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L):
        parameters["W"+str(l)] -= learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate*grads["db"+str(l)]
    return parameters


def neural_net(X, y, layer_dims, learning_rate, num_iterations= 2000, print_cost = True):
    parameters = initialize_parameters(layer_dims)
    costs = []
    accuracies = []
    for i in range(0, num_iterations):
        AL, caches = forward_pass(X, parameters)
        cost = compute_cost(AL, y)
        _, accuracy = predict(X, y, parameters)
        grads = backward_pass(AL, y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            # print ("Cost after iteration %i: %f" %(i, cost))
            print("*****************************************************")
            print(f"iteration {i+1}: loss: {cost}, accuracy: {accuracy}\n")
        costs.append(cost)
        accuracies.append(accuracy)

    return parameters, costs, accuracies, learning_rate 

def plot_history(costs, accuracies, lr):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))

    plt.subplot(221)
    plt.plot(np.squeeze(costs), color="red", label="loss")
    plt.ylim(0, 1)
    plt.xlabel("cost")
    plt.ylabel("iterations (per hundreds)")
    plt.legend()
    plt.title("Learning rate =" + str(lr))

    plt.subplot(222)
    plt.plot(np.squeeze(accuracies), color="blue", label="accuracy")
    plt.ylim(0,1)
    plt.xlabel("accuracy")
    plt.ylabel("iterations (per hundreds)")
    plt.legend()
    plt.title("Learning rate =" + str(lr))

    plt.subplot(223)
    plt.plot(np.squeeze(costs), color="red", label="loss")
    plt.plot(np.squeeze(accuracies), color="blue", label="accuracy")
    plt.ylim(0,1)
    plt.ylabel('cost and accuracy')
    plt.xlabel('iterations (per hundreds)')
    plt.legend()
    plt.title("Learning rate =" + str(lr))

    plt.tight_layout()
    plt.show()



def predict(X, y, params):
    m = X.shape[1]
    n = len(params)
    p = np.zeros((1, m))
    probabs, _ = forward_pass(X, params)
    for i in range(0, probabs.shape[1]):
        if probabs[0, i] >= 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    accuracy = np.sum((p == y)/m)
    # print ("true labels: " + str(y))
    # print ("predictions: " + str(p))
    # print("Accuracy: "  + str(accuracy))
    return p, accuracy




if __name__=="__main__":
    # X, parameters = L_model_forward_test_case()
    # AL, caches = forward_pass(X, parameters)
    # print(AL)
    import matplotlib.pyplot as plt
    np.random.seed(1)
    layer_dims = [3, 20, 10, 10, 1]
    X = np.random.randn(3, 100)
    Y = np.random.choice([0,1], size=(1, 100))
    parameters, costs, accuracy, lr = neural_net(X, Y, layer_dims, 0.1, num_iterations=2000, print_cost= True)
    p, _ = predict(X, Y, parameters)
    # print(Y)
    # print(p, _)
    plot_history(costs, accuracy, lr)
    



    




    

