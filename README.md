# nnarch

`nnarch` is a collection of AI (Classical, ML, DL) algorithms implementations in python. It is bundled as a command line application with `train` and `infer` commands for each model architecture. The training pipeline also includes dataset loading and preparation. Look at [ai-notebooks](https://github.com/HannesKimara/ai-notebooks) repo for readable source implementations.


Currently supports training(from scratch/pickled checkpoints) and inference of a multi-layer perceptron(MLP) on the MNIST reference dataset.

## Getting started

To train the MLP on the MNIST dataset clone this repository and run:

```python
python -m nnarch mlp train
```

This trains a model with 3 linear layers and 2 ReLu activation layers defined with the following architecture and yielding a validation accuracy of about 97.6% by default. The best achieved with a multi step train is about 98.3%

```python
model = Model([
        Linear(input_shape, 128), # input_shape = 784
        ReLu(),
        Linear(128, 64),
        ReLu(),
        Linear(64, num_classes) # num_classes = 10
    ])
```


## Known Issues

The MLP implementation in numpy has numerical instability for large learning rates and broadcast errors for large batch sizes(>=64).

## License

This repository is licenses under the [`Apache License`](./LICENSE) license