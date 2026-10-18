from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import hashlib, os
from pathlib import Path
from mstar_v1_common import atomic_json, digest, file_hash, read_json

def selection_identity(args, record):
    r = record['result']
    if r['status'] != 'complete':
        raise ValueError('Only a complete validation-selected trial is eligible')
    return {'schema': 'MSTAR_original6_best_model_v1', 'protocol_id': args.protocol_id, 'dataset_id': args.dataset_id, 'algorithm': args.algorithm, 'run': args.run, 'selected_fe': record['fe'], 'candidate_id': record['candidate_id'], 'config': r['config'], 'training_seed': r['training_seed'], 'best_epoch': r['best_epoch'], 'validation_macro_f1': r['f1'], 'validation_loss': r['loss']}

def state_digest(state):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError('Checkpoint must contain tensor state_dict values')
        value = value.detach().cpu().contiguous()
        h.update(digest([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def save_best(directory, identity, state):
    import torch
    if not state:
        raise ValueError('Missing best-epoch model weights')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob('best_model.*.tmp'):
        stale.unlink(missing_ok=True)
    path = directory / 'best_model.pt'
    tmp = directory / f'best_model.{os.getpid()}.tmp'
    state = {k: v.detach().cpu().clone() for k, v in state.items()}
    payload = {'selection': identity, 'state_dict': state, 'state_sha256': state_digest(state)}
    try:
        with tmp.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    atomic_json(directory / 'best_model.json', {'selection': identity, 'weights_sha256': file_hash(path), 'state_sha256': payload['state_sha256'], 'file': 'best_model.pt'})

def load_best(directory, expected, allowed_selections=()):
    import torch
    directory = Path(directory)
    path = directory / 'best_model.pt'
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Unreadable best checkpoint; do not silently accept corruption') from exc
    if not isinstance(payload, dict) or set(payload) != {'selection', 'state_dict', 'state_sha256'}:
        raise ValueError('Invalid checkpoint format')
    identity = payload['selection']
    if payload['state_sha256'] != state_digest(payload['state_dict']):
        raise ValueError('Checkpoint tensor checksum differs')
    if identity != expected and identity not in allowed_selections:
        raise ValueError('Checkpoint configuration/seed/epoch/track differs from committed journal')
    meta_path = directory / 'best_model.json'
    meta = read_json(meta_path) if meta_path.exists() else None
    if meta and meta.get('selection') == identity and (meta.get('weights_sha256') != file_hash(path)):
        raise ValueError('Checkpoint file checksum differs')
    if not meta or meta.get('selection') != identity:
        atomic_json(meta_path, {'selection': identity, 'weights_sha256': file_hash(path), 'state_sha256': payload['state_sha256'], 'file': 'best_model.pt'})
    return payload if identity == expected else None
