import numpy as np
import matplotlib.pyplot as plt


def relu_(z):
    return z * (z > 0)


def relu(z):
    return relu_(z), z


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))


def sigmoid(z):
    return sigmoid_(z), z


def d_relu(z):
    return 1 * (z >= 0)


def d_sigmoid(z):
    return sigmoid_(z) * (1 - sigmoid_(z))


def backward_relu(dA, cache_activation):
    return dA * d_relu(cache_activation)


def backward_sigmoid(dA, cache_activation):
    return dA * d_sigmoid(cache_activation)


# Initialize weights and biases
def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for i in range(1, L):
        parameters["W" + str(i)] = (
            np.random.randn(layers_dims[i], layers_dims[i - 1])
            * (2 / layers_dims[i - 1]) ** 0.5
        )
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


# Update weights based on gradient
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for i in range(L):
        parameters["W" + str(i + 1)] = (
            parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        )
        parameters["b" + str(i + 1)] = (
            parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
        )

    return parameters


# Compute Z
def forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# Compute A
def forward_activation(A_prev, W, b, activation_function):
    Z, cache_linear = forward_linear(A_prev, W, b)

    if activation_function == "relu":
        A, cache_activation = relu(Z)

    elif activation_function == "sigmoid":
        A, cache_activation = sigmoid(Z)

    return A, (cache_linear, cache_activation)


# Complete one full forward pass
def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        A, cache = forward_activation(
            A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu"
        )
        caches.append(cache)
    AL, cache = forward_activation(
        A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
    )
    caches.append(cache)
    return AL, caches


def compute_L2_regularization(m, parameters, lambd):
    L = len(parameters) // 2
    W_sum = 0
    for i in range(1, L):
        W_sum += np.sum(np.square(parameters["W" + str(i)]))

    return (1 / m) * (lambd / 2) * W_sum


def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL.T)))

    if lambd != 0:
        cost += compute_L2_regularization(m, parameters, lambd)

    return np.squeeze(cost)


# Compute derivatives
def backward_linear(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# Compute activation derivatives
def backward_activation(dA, cache, activation_function):
    cache_linear, cache_activation = cache

    if activation_function == "relu":
        dZ = backward_relu(dA, cache_activation)
    elif activation_function == "sigmoid":
        dZ = backward_sigmoid(dA, cache_activation)

    dA_prev, dW, db = backward_linear(dZ, cache_linear)

    return dA_prev, dW, db


# Complete one full backward pass
def backward_propagation(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cache = caches[L - 1]
    (
        grads["dA" + str(L - 1)],
        grads["dW" + str(L)],
        grads["db" + str(L)],
    ) = backward_activation(dAL, cache, "sigmoid")

    cache = caches[L - 1]

    for i in reversed(range(L - 1)):
        cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = backward_activation(
            grads["dA" + str(i + 1)], cache, "relu"
        )
        grads["dA" + str(i)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    if lambd != 0:
        for i in range(L - 1):
            cache = caches[i][0]
            _, W, _ = cache
            grads["dW" + str(i + 1)] = grads["dW" + str(i + 1)] + (lambd / m) * W

    return grads


# create a neural net
def model(
    X,
    Y,
    layers_dims,
    learning_rate=0.01,
    num_iterations=1000,
    print_cost=False,
    lambd=0,
):
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y, parameters, lambd)
        grads = backward_propagation(AL, Y, caches, lambd)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# Make predictions given a model
def predict(parameters, X):
    AL, _ = forward_propagation(X, parameters)
    predictions = AL >= 0.5
    return predictions
