from __future__ import annotations
import csv, hashlib, json, os, random
from pathlib import Path
VERSION = '2.0.0-mstar-original-six'
TOTAL_CONFIGS = 3 ** 8

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_csv(path, rows, fields=None):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def seed_for(*parts):
    return int(digest(list(parts))[:8], 16) % (2 ** 31 - 1)

def candidate_order(seed, run):
    order = list(range(TOTAL_CONFIGS))
    random.Random(seed_for('RandomSearch', seed, run)).shuffle(order)
    return order

def genes_from_id(cid):
    if not 0 <= cid < TOTAL_CONFIGS:
        raise ValueError('candidate_id outside 0..6560')
    result = [0] * 8
    for i in range(7, -1, -1):
        result[i] = cid % 3
        cid //= 3
    return result

def decode_candidate(cid):
    g = genes_from_id(cid)
    names = ['optimizer', 'learning_rate', 'weight_decay', 'batch_size', 'epochs', 'augmentation_strength', 'scheduler', 'scheduler_level']
    values = [['SGD', 'Adam', 'RMSprop'], [0.01, 0.001, 0.0001], [0.01, 0.001, 0.0001], [32, 64, 16], [20, 30, 10], [0.5, 1.0, 1.5], ['StepLR', 'CosineAnnealingLR', 'ExponentialLR'], [0, 1, 2]]
    cfg = {n: vs[i] for n, vs, i in zip(names, values, g)}
    level = cfg['scheduler_level']
    cfg['scheduler_params'] = {'StepLR': {'gamma': [0.1, 0.5, 0.9][level], 'step_size': [5, 10, 15][level]}, 'CosineAnnealingLR': {'eta_min': [1e-06, 1e-05, 0.0001][level], 'T_max': [10, 20, 30][level]}, 'ExponentialLR': {'gamma': [0.88, 0.93, 0.98][level]}}[cfg['scheduler']]
    return {'candidate_id': cid, 'genes': g, **cfg}

def parse_runs(text, total):
    result = []
    for token in text.split(','):
        if '-' in token:
            start, end = map(int, token.split('-', 1))
            if end < start:
                raise ValueError('reversed run range')
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise ValueError('run IDs must be nonempty and distinct')
    if any((r < 0 or r >= total for r in result)):
        raise ValueError(f'run IDs must be in 0..{total - 1}')
    return sorted(result)

def runtime_hashes():
    root = Path(__file__).parent
    names = ['algorithm_factory.py', 'legacy_baselines.py', 'ga_basic.py', 'de_optimizers.py', 'journal.py', 'common.py']
    return {n: file_hash(root / n) for n in names}

class FileLock:

    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0)
        if self.file.read(1) == b'':
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError(f'Another process is using {self.path.parent}')
        return self

    def __exit__(self, *args):
        if self.file:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
