from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from pathlib import Path
import numpy as np
from PIL import Image

def read_sample_uint8(path):
    with Image.open(Path(path)) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError(f'Multi-frame image is not supported: {path}')
        array = np.asarray(im)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError(f'Expected single-channel uint8, got {array.dtype}/{array.shape}: {path}')
        if array.shape != (128, 128):
            raise ValueError(f'Expected 128x128 chip, got {array.shape}: {path}')
        if array.max() == array.min():
            raise ValueError(f'Constant image: {path}')
        return np.array(array, copy=True)
