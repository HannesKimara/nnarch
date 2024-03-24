import os
import tempfile


TEMP_DIR = tempfile.gettempdir()
NNARCH_MAGIC = "NNARCH_DATADIR"

DATA_DIR = os.path.join(TEMP_DIR, NNARCH_MAGIC)
