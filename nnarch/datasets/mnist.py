import os
import gzip
import hashlib
import tempfile

import numpy as np
import requests
from tqdm import tqdm

from nnarch.datasets.config import DATA_DIR


def fetch_mnist(url: str, dir_label='mnist'):
    """
    Loads mnist-like(gzip compressed) dataset from the link, decompresses and returns a numpy array of data type `uint8`.
    The datasets are cached in the tempdir with file names encoded as the md5 hash of the url

    :param url: url of the dataset to load
    """
    base_dir = os.path.join(DATA_DIR, dir_label)
    tmp_path = os.path.join(base_dir, hashlib.md5(url.encode('utf-8')).hexdigest())
    
    if os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:
            data = f.read()
    else:
        os.makedirs(base_dir, exist_ok=True)
        res = requests.get(url, stream=True)
        total = res.headers.get('content-length')
        
        if total is None:
            total = 0
        
        total = int(total)
        
        block_size = 1024
        progress_bar = tqdm(total=total, unit='iB', unit_scale=True)
        
        with open(tmp_path, 'wb+') as f:
            for iter_data in res.iter_content(block_size):
                progress_bar.update(len(iter_data))
                f.write(iter_data)
            data = f.read()
        progress_bar.close()

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

def to_categorical(y, num_classes=10):
    """
    performs one-hot encoding on y

    :param y: a numpy-like array
    :param num_classes: the number of classes in the dataset. Defaults to 10
    """
    return np.eye(num_classes)[y]
