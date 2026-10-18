from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from pathlib import Path
import json, os, shutil
import numpy as np
from mstar_common import FileLock, atomic_json, digest, file_hash, read_json

def build_cache(samples, destination, dataset_id, progress=None):
    import torch, torchvision
    from torchvision import transforms
    from PIL import Image
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    identity = {'dataset_id': dataset_id, 'torchvision': torchvision.__version__, 'operation': 'PIL RGB Resize224 default -> ToTensor FP32 v1', 'paths': [str(Path(p).resolve()) for p, _ in samples]}
    folder = root / digest(identity)
    with FileLock(root / 'tensor_build.lock'):
        if folder.exists():
            open_cache(folder, dataset_id, verify=True)
            return str(folder)
        stage = folder.with_name(folder.name + '.building')
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        try:
            array = np.lib.format.open_memmap(stage / 'images.npy', mode='w+', dtype=np.float32, shape=(len(samples), 3, 224, 224))
            resize = transforms.Resize((224, 224))
            tensor = transforms.ToTensor()
            for i, (path, _) in enumerate(samples):
                with Image.open(path) as raw:
                    x = tensor(resize(raw.convert('RGB')))
                array[i] = x.numpy()
                if progress and (i % 64 == 0 or i + 1 == len(samples)):
                    progress((i + 1) / len(samples))
            array.flush()
            del array
            atomic_json(stage / 'manifest.json', {'identity': identity, 'sha256': file_hash(stage / 'images.npy')})
            stage.replace(folder)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return str(folder)

class TensorCache:

    def __init__(self, folder, dataset_id, verify=False):
        folder = Path(folder)
        meta = read_json(folder / 'manifest.json')
        if meta['identity']['dataset_id'] != dataset_id:
            raise ValueError('Tensor cache dataset mismatch')
        if folder.name != digest(meta['identity']):
            raise ValueError('Tensor cache identity mismatch')
        path = folder / 'images.npy'
        if verify and file_hash(path) != meta['sha256']:
            raise ValueError('Tensor cache checksum mismatch')
        self.array = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.array.dtype != np.float32 or self.array.shape != (len(meta['identity']['paths']), 3, 224, 224):
            raise ValueError('Tensor cache shape/dtype mismatch')
        self.indices = {p: i for i, p in enumerate(meta['identity']['paths'])}
        self.path = path
        self.stamp = (path.stat().st_size, path.stat().st_mtime_ns)

    def get(self, path):
        import torch
        if (self.path.stat().st_size, self.path.stat().st_mtime_ns) != self.stamp:
            raise ValueError('Tensor cache changed during execution')
        i = self.indices.get(str(Path(path).resolve()))
        if i is None:
            return None
        return torch.from_numpy(np.array(self.array[i], copy=True))

def open_cache(folder, dataset_id, verify=False):
    return TensorCache(folder, dataset_id, verify)
