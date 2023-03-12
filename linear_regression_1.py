import numpy as np
import torch


def model(x):   # Multiples weights with inputs and adds biases to predict output
    return x @ w.t() + b  # @ shows that it is the dot product, .t() gets the transpose of the matrix


def mse(t1, t2):    # returns the mean square error, numel = no. of elements
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Conversion of numpy array to pytorch tensor
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

print(inputs)
print(targets)

# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)

preds = model(inputs)
print(preds)
print(targets)

loss = mse(preds, targets)
print(loss)

loss.backward()
print(w.grad)
print(b.grad)

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

print(w)
print(b)

preds = model(inputs)
loss = mse(preds, targets)
print(loss)

for i in range(1000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print(loss)

print(preds)
print(targets)
