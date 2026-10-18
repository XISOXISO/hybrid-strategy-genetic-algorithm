from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import os, time, urllib.request
from pathlib import Path
import torch
from mstar_common import file_hash
URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'

def main():
    path = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not file_hash(path).startswith('f37072fd'):
            raise ValueError('Existing initial weights checksum mismatch')
        print('Initial weights already verified:', path)
        return
    tmp = path.with_suffix('.downloading')
    url = os.environ.get('WEIGHTS_URL', URL)
    for attempt in range(1, 5):
        print(f'ImageNet weights download {attempt}/4', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as target:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            if not file_hash(tmp).startswith('f37072fd'):
                raise ValueError('Initial weights checksum mismatch')
            tmp.replace(path)
            print('Verified:', path)
            return
        except ValueError:
            tmp.unlink(missing_ok=True)
            raise
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 * attempt)
if __name__ == '__main__':
    main()
