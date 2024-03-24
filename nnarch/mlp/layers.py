import numpy as np


class ReLu:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x):
        self.x = x
        return np.maximum(0, self.x)
        
    def backward(self, grad):
        return 1 * (self.x > 0) * grad

    def __call__(self, x):
        return self.forward(x)


class LeakyReLu:
    def __init__(self, k=0.01):
        self.params = {}
        self.grads = {}
        self.k = k

    def forward(self, x):
        self.x = x
        return (self.x > 0) * self.x + (self.x < 0) * self.x * self.k

    def backward(self, grad):
        return (self.x > 0) * grad + (self.x < 0) * grad * self.k 

    def __call__(self, x):
        return self.forward(x)


class Linear:
    def __init__(self, input_size, output_size, bias=True):
        self.params = {}
        self.grads = {}
        self.params["w"] = np.random.uniform(-1., 1., size=(input_size, output_size))/np.sqrt(input_size*output_size)

        if bias:
            self.params["b"] = np.random.uniform(-1., 1., size=(output_size))/np.sqrt(output_size)

        else:
            self.params['b'] = np.zeros_like(output_size, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.params['w']) + self.params['b']
    
    def backward(self, grad):
        self.grads['b'] = np.sum(grad)
        self.grads['w'] = self.x.T @ grad 

        return grad @ self.params['w'].T

    def __call__(self, inputs):
        return self.forward(inputs)
