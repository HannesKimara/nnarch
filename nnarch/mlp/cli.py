import os
import pickle

import click
import numpy as np
from PIL import Image

from nnarch.mlp.model import Model
from nnarch.mlp.layers import Linear, ReLu, LeakyReLu
from nnarch.mlp.train import BatchIterator, SGD
from nnarch.datasets.mnist import fetch_mnist, to_categorical


def init_model(input_shape: int, num_classes: int=10) -> Model:
    model = Model([
        Linear(input_shape, 128),
        ReLu(),
        Linear(128, 64),
        ReLu(),
        Linear(64, num_classes)
    ])

    return model


@click.group()
def mlp():
    """
    Multi-Layer Perceptron training and inference utility
    """
    pass


@mlp.command()
@click.option('--epochs', default=10, help='Number of epochs for training.')
@click.option('--lr', default=1e-3, help='Learning rate for training.')
@click.option('--batch-size', default=32, help='Batch size for training.')
@click.option('--show-history', default=False, help='Print history plot')
@click.option('--out-model', default='model.pkl', help='Model output path')
@click.option('--from-checkpoint', default='', help='trains from a pickled model as checkpoint')
def train(epochs: int, lr: float, batch_size: int, show_history: bool, out_model: str, from_checkpoint: str):
    """
    loads the mnist dataset and trains the model

    :param epochs: the number of epochs to train on
    :param lr: the learning rate
    :param batch_size: the batch size to train on every step 
    :param out_model: the path to export the pickled model after training
    """
    X_train = fetch_mnist('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[16:].reshape(-1, 28, 28)
    Y_train = fetch_mnist('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[8:]
    X_test = fetch_mnist('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[16:].reshape(-1, 28, 28)
    Y_test = fetch_mnist('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[8:]

    X_train = (X_train / 255.).reshape((-1, 28 * 28))
    X_test = (X_test / 255.).reshape((-1, 28 * 28))

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Init the multi layer perceptron with flattened input shape
    input_shape = np.product(X_train[0].shape)
    
    if from_checkpoint != '':
        with open(from_checkpoint, 'rb') as f:
            mlp = pickle.load(f)
    else:
        mlp = init_model(input_shape)
    
    # Init the optimizer
    optimizer = SGD(lr=lr)

    _ = mlp.fit(
        BatchIterator(X_train, Y_train, batch_size=batch_size),
        epochs=epochs, 
        validation_data=(X_test, Y_test),
        optim=optimizer
    )

    # Save the model
    with open(out_model, 'wb') as f:
        pickle.dump(mlp, f)

    click.echo(f'\nTraining completed with epochs={epochs}, lr={lr}, batch_size={batch_size}')


@mlp.command()
@click.option('--model-path', default='model.pkl', help='Path to the pickled model')
@click.option('--image-path', help='Path to the image file for inference', required=True)
def infer(model_path: str, image_path: str):
    """
    loads the model and runs inference on image provided

    :param model_path: the path to the pickled model, This defaults to `model.pkl`
    :param image_path: the path to the image to run inference on
    """
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    raw_im = Image.open(image_path)

    im_arr = np.array(raw_im.resize((28, 28)))
    im_arr = im_arr / 255.

    input_data = im_arr.reshape(-1, 28 * 28)
    result = model.forward(input_data)    

    click.echo(f'Prediction : {np.argmax(result)}')
