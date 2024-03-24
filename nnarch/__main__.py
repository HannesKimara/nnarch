import click


from nnarch.mlp.cli import mlp
from nnarch.datasets.cli import datasets
from nnarch.transformers.cli import transformers


@click.group()
def nnarch_cli():
    """
    Train, benchmark and run inference on multiple architectures on different reference datasets.
    """
    pass


if __name__ == '__main__':
    nnarch_cli.add_command(mlp)
    nnarch_cli.add_command(datasets)
    nnarch_cli.add_command(transformers)
    
    nnarch_cli()
