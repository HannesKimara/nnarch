import numpy as np
from tqdm import tqdm

from nnarch.mlp.errors import MSE
from nnarch.mlp.train import SGD


class Model:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
    
    def fit(self, batch_dataset, epochs = 10, loss = MSE(), optim = SGD(), validation_data=()):
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_iterator = tqdm(batch_dataset(), total=batch_dataset.batch_iter_count, unit=' batch(s)')
            for idx, batch in enumerate(batch_iterator):
                yHat = self.forward(batch[0])
                epoch_loss += loss.loss(yHat, batch[-1])
                grad = loss.grad(yHat, batch[-1])

                self.backward(grad)
                optim.step(self)
                batch_iterator.set_description(f"Epoch {epoch + 1}/{epochs}")
                ## batch_iterator.set_postfix({f"batch_accuracy": pvalue(batch[0], batch[-1])})
                
                if idx == batch_dataset.batch_iter_count - 1:
                    val_loss = np.average(loss.loss(self.forward(validation_data[0]), validation_data[-1]))
                    val_accuracy = self.evaluate(*validation_data)
                    batch_iterator.set_postfix({
                        'val_accuracy': val_accuracy,
                        'val_loss': val_loss
                })
                    
            epoch_loss /= idx
            history['train_loss'].append(epoch_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_loss'].append(val_loss)
        
        history['train_loss'] = np.average(np.array(history['train_loss']), axis=1)
        history['val_accuracy'] = np.array(history['val_accuracy'])
                
        return history
    
    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def __call__(self, x):
        return self.forward(x)
    
    def evaluate(self, x_test, y):
        return evaluate_model(self, x_test, y)
    
def evaluate_model(model: Model, x_test, y):
    y_mega_hat = model.forward(x_test)

    preds = np.array([np.argmax(i) for i in y_mega_hat])
    actuals = np.array([np.argmax(i) for i in y])
    simi = preds == actuals
    
    return np.count_nonzero(simi)/len(simi)
