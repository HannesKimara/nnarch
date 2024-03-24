import click


@click.group()
def transformers():
    """
    Transformers training and inference utility
    """
    pass


@transformers.command()
def train():
    """
    train the model
    """
    raise NotImplementedError

@transformers.command()
def infer():
    """
    run inference against the model
    """
    raise NotImplementedError