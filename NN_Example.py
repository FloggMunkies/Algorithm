import numpy as np

np.set_printoptions(suppress=True)

# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
x = np.array ([ [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1] ])
print(x.shape)
# output dataset
y = np.array([[0, 1, 0, 1]]).T

# seed random numbers to make calculation  deterministic
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(1000):

    # forward propagation
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    # error

    l1_error = y - l1

    # multiply error by slope of sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1, True)

    print(l0.T.shape)
    print(l1_delta.shape)
    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output after Training:")
print(l1)

