from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from contextlib import contextmanager
import json, os, sqlite3
from pathlib import Path
from mstar_v1_common import atomic_json, digest, read_json

class CacheBusy(BaseException):

    def __init__(self, path):
        self.path = str(path)
        super().__init__('Candidate is being evaluated: ' + self.path)

@contextmanager
def candidate_lock(path, blocking=False):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise CacheBusy(path)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def available(path):
    try:
        with candidate_lock(path):
            return True
    except CacheBusy:
        return False

def weight_identity(args, run, record):
    r = record['result']
    return dict(schema='shared-best-v4', cache_id=args.cache_id, dataset_id=args.dataset_id, run=run, candidate_id=record['candidate_id'], config=r['config'], training_seed=r['training_seed'], best_epoch=r['best_epoch'], validation_macro_f1=r['f1'], validation_loss=r['loss'])

class SharedWeights:

    def __init__(self, args):
        self.args = args
        self.root = Path(args.output) / 'shared_weights'
        self.root.mkdir(parents=True, exist_ok=True)

    def _refs(self):
        p = self.root / 'references.json'
        if not p.exists():
            return {}
        obj = read_json(p)
        refs = obj['references']
        if obj['checksum'] != digest(refs):
            raise ValueError('Shared-weight references checksum changed')
        return refs

    def _write(self, refs):
        atomic_json(self.root / 'references.json', dict(references=refs, checksum=digest(refs)))

    def _folder(self, identity):
        return self.root / 'objects' / digest(identity)

    def _prune(self, refs):
        import shutil
        keep = set(refs.values())
        for folder in (self.root / 'objects').glob('*'):
            if folder.is_dir() and folder.name not in keep:
                shutil.rmtree(folder)

    def get(self, identity, lease=None):
        from mstar_v1_checkpoints import load_best
        with candidate_lock(self.root / 'index.lock', True):
            folder = self._folder(identity)
            if not (folder / 'best_model.pt').exists():
                return None
            payload = load_best(folder, identity)
            if lease:
                refs = self._refs()
                refs[lease] = folder.name
                self._write(refs)
                self._prune(refs)
            return payload['state_dict']

    def publish(self, identity, state, lease):
        from mstar_v1_checkpoints import load_best, save_best, state_digest
        with candidate_lock(self.root / 'index.lock', True):
            folder = self._folder(identity)
            if (folder / 'best_model.pt').exists():
                old = load_best(folder, identity)
                if state_digest(old['state_dict']) != state_digest(state):
                    raise ValueError('Same deterministic candidate produced different weights')
            else:
                save_best(folder, identity, state)
            refs = self._refs()
            refs[lease] = folder.name
            self._write(refs)
            self._prune(refs)

    def pin(self, identity, lease):
        with candidate_lock(self.root / 'index.lock', True):
            refs = self._refs()
            refs[lease] = digest(identity)
            self._write(refs)
            self._prune(refs)
