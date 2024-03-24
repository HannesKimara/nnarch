import numpy as np


class MSE:
    # Not really Mean Squared Error
    def loss(self, yHat, y):
        return np.sum((yHat - y)**2, axis=1)
    
    def grad(self, yHat, y):
        return 2 * (yHat - y)

    def __call__(self, yHat, y):
        return self.loss(yHat, y)