from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from collections import Counter
from pathlib import Path
from PIL import Image
from mstar_v1_common import digest, file_hash, read_json
ROLES = ('train', 'validation', 'test')

def safe_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe manifest path')
    out = (root / p).resolve()
    if root not in out.parents:
        raise ValueError('Manifest path escapes data root')
    return out

def validate_manifest(data, expected_dataset_id=None, exact_default=True):
    identity = data['identity']
    rows = identity['samples']
    classes = identity['classes']
    if digest(identity) != data['dataset_id']:
        raise ValueError('Frozen dataset checksum changed')
    if expected_dataset_id is not None and data['dataset_id'] != expected_dataset_id:
        raise ValueError('Dataset ID differs from original frozen split')
    if len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError('Expected ten classes')
    counts = Counter()
    seen = set()
    for row in rows:
        role, label = (row['split'], row['label'])
        path = Path(row['copied_image'])
        if role not in ROLES or type(label) != int or (not 0 <= label < 10) or (row['class'] != classes[label]):
            raise ValueError('Invalid frozen role/class/label')
        if path.is_absolute() or '..' in path.parts or str(path.parent.name) != str(label):
            raise ValueError('Invalid frozen image path/label')
        if str(path) in seen:
            raise ValueError('Overlapping frozen split paths')
        seen.add(str(path))
        counts[role] += 1
        for field in ('sha256', 'decoded_sha256'):
            if len(row[field]) != 64 or any((c not in '0123456789abcdef' for c in row[field])):
                raise ValueError('Malformed image checksum')
    if dict(counts) != identity['split_counts']:
        raise ValueError('Frozen split counts differ')
    if exact_default and dict(counts) != {'train': 3617, 'validation': 776, 'test': 779}:
        raise ValueError('Expected original 3617/776/779 membership')
    return data

def check_image(folder, row):
    p = safe_path(folder, row['copied_image'])
    if file_hash(p) != row['sha256']:
        raise ValueError('Image integrity changed: ' + str(p))
    import hashlib
    with Image.open(p) as im:
        im.load()
        pixels = im.convert('RGB').tobytes()
    if hashlib.sha256(pixels).hexdigest() != row['decoded_sha256']:
        raise ValueError('Decoded image pixels changed')
    return p

def load_prepared(folder, root_override=None, verify_train_val=False, progress=None, expected_dataset_id=None, image_source='copied', exact_default=True):
    folder = Path(folder).resolve()
    data = validate_manifest(read_json(folder / 'dataset.json'), expected_dataset_id, exact_default)
    samples = {'train': [], 'validation': []}
    rows = [r for r in data['identity']['samples'] if r['split'] in samples]
    for i, row in enumerate(rows, 1):
        p = check_image(folder, row) if verify_train_val else safe_path(folder, row['copied_image'])
        samples[row['split']].append((str(p), row['label']))
        if progress and (i % 50 == 0 or i == len(rows)):
            progress(i / len(rows), f'Checked train/val {i}/{len(rows)}; test untouched')
    return (data, samples)
