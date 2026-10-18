from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import os, shutil
from pathlib import Path
from mstar_common import atomic_json, digest, file_hash, read_json
from src.HGA.mstar.checkpoints import selection_identity, load_best
from report_mstar import selected

def budgets(max_fe):
    return sorted(set(range(50, max_fe + 1, 50)) | {max_fe})

def snapshot(args, directory, rows, upto_fe):
    rd = Path(directory)
    archive = rd / 'budget_checkpoints'
    records = []
    for fe in budgets(args.max_fe):
        if fe > upto_fe:
            continue
        best = selected(rows, fe)
        if best is None:
            continue
        expected = selection_identity(args, best)
        key = digest(expected)
        obj = archive / 'objects' / key
        index = archive / f'fe_{fe:04d}.json'
        if index.exists():
            meta = read_json(index)
            check = dict(meta)
            checksum = check.pop('checksum', None)
            if checksum != digest(check) or meta['budget_fe'] != fe or meta['selection'] != expected:
                raise ValueError('Budget archive selection/checksum mismatch')
            if meta['status'] == 'available' and (not (obj / 'best_model.pt').exists()):
                raise ValueError('Previously archived checkpoint is missing')
        available = False
        if (obj / 'best_model.pt').exists():
            load_best(obj, expected)
            available = True
        elif (rd / 'best_model.json').exists() and read_json(rd / 'best_model.json')['selection'] == expected:
            load_best(rd, expected)
            obj.mkdir(parents=True, exist_ok=True)
            try:
                os.link(rd / 'best_model.pt', obj / 'best_model.pt')
            except OSError:
                shutil.copy2(rd / 'best_model.pt', obj / 'best_model.pt')
            shutil.copy2(rd / 'best_model.json', obj / 'best_model.json')
            load_best(obj, expected)
            available = True
        meta = {'budget_fe': fe, 'selection': expected, 'object_key': key, 'status': 'available' if available else 'historical_weights_missing', 'weights_sha256': file_hash(obj / 'best_model.pt') if available else None, 'policy': 'validation-selected best within fixed FE budget; offline tests have no search feedback'}
        meta['checksum'] = digest(meta)
        atomic_json(index, meta)
        records.append(meta)
    return records

def validate_entry(directory, index, args, rows, require_weights=True):
    rd = Path(directory)
    meta = read_json(index)
    c = dict(meta)
    checksum = c.pop('checksum', None)
    if checksum != digest(c):
        raise ValueError('Budget archive metadata checksum mismatch')
    fe = meta['budget_fe']
    if fe not in budgets(args.max_fe) or fe > len(rows):
        raise ValueError('Invalid archived FE budget')
    best = selected(rows, fe)
    if best is None or meta['selection'] != selection_identity(args, best) or meta['object_key'] != digest(meta['selection']):
        raise ValueError('Archived selection differs from committed validation history')
    obj = rd / 'budget_checkpoints' / 'objects' / meta['object_key']
    if meta['status'] == 'available':
        if require_weights or (obj / 'best_model.pt').exists():
            if file_hash(obj / 'best_model.pt') != meta['weights_sha256']:
                raise ValueError('Archived weights checksum mismatch')
            load_best(obj, meta['selection'])
    elif meta['status'] != 'historical_weights_missing':
        raise ValueError('Invalid archive status')
    return (meta, obj)
