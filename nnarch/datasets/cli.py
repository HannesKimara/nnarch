import os
import shutil
from pprint import pprint

import click

from nnarch.datasets.config import DATA_DIR


@click.group()
def datasets():
    """
    Dataset management utility
    """
    pass


@datasets.command()
def list():
    """
    List cached datasets by label
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    datasets = os.listdir(DATA_DIR)
    if len(datasets) == 0:
        click.echo('No datasets cached yet')
        return 

    click.echo(f'Datasets ({len(datasets)}):\n')
    for ds in datasets:
        click.echo(f'- {ds} ')


@datasets.command()
@click.option('--label', required=True, help='Clean dataset by label. Use `nnarch datasets list` to get dataset labels')
def remove(label: str):
    """
    Clean cached datasets
    """
    label_dir = os.path.join(DATA_DIR, label)

    if not os.path.exists(label_dir):
        click.echo(f'No dataset with label `{label}` found. Use `nnarch datasets list` to get dataset labels\n')
        return
    
    shutil.rmtree(label_dir)

    click.echo(f'Successfully removed - {label}\n')
