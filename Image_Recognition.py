from PIL import Image
import numpy as np


# Functions

def load_input(folder, filename, greyscale=True):

    if greyscale:
        i = Image.open(folder + "/" + filename).convert("L")

    i = Image.open(folder +"/" + filename).convert("L")

    print("Loading file: " + filename)
    return np.array(i)

def load_multiple(folder, filetype, amount, start=0, greyscale=True):
    a = np.array([])
    b = []

    for i in range(amount-start):
        a = load_input(folder, str(i) + "." + filetype)
        b.append(a.flatten())

    b = np.asarray(b)
    return b

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input

number_inputs = 21
x = load_multiple("Asteroid_0", "png", number_inputs)
# output
y = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T


# random seed
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((2880, 1)) - 1

for iter in range(1000):

    # forward propagation
    for i in range(number_inputs):
        l0 = x[i]
        l1 = nonlin(np.dot(l0, syn0))

        # error

        l1_error = y - l1

        # multiply error by slope of sigmoid at values in l1
        l1_delta = l1_error * nonlin(l1, True)

        # update weights
        syn0 += np.dot(l0.T, l1_delta)

print("Output after Training:")
print(l1)