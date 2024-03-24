import numpy as np


class BatchIterator:
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.batch_iter_count = round(len(self.x) / self.batch_size)
        
    def __call__(self):
        assert len(self.x) == len(self.y)
        starts = np.arange(0, len(self.x), self.batch_size)
        
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.batch_size
            batch_x = self.x[start:end]
            batch_y = self.y[start:end]
            
            yield (batch_x, batch_y)


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr
        
    def step(self, model):
        for param, grad in model.get_params_and_grads():
            param -= self.lr * grad