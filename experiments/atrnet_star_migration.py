from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import copy, json, os, shutil, sqlite3, time, tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from atrnet_star_common import FileLock, atomic_json, digest, file_hash, read_json, decode_candidate, seed_for
from src.HGA.atrnet_star.journal import Journal, read_events, hash_json
from report_atrnet_star import collect, selected
from src.HGA.atrnet_star.checkpoints import selection_identity
from atrnet_star_final_evaluation import TEST_POLICY, validate_test_result
LEGACY = ['RandomSearch', 'TPE', 'CMA_ES', 'GridSearchWithBudget', 'ASHA', 'EvolutionaryHPO', 'MomentumBasedRandomSearch', 'AdamHPO', 'GWO', 'PSO', 'GA_Basic']
HGA = ['HGA_RM', 'HGA_CM', 'HGA_SW50', 'HGA_SW60']

def check_protocol(source, current):
    p = read_json(source / 'protocol.json')
    identity = p['identity']
    b = identity['benchmark']
    new = current['identity']['benchmark']
    if digest(identity) != p['protocol_id'] or digest(b) != identity['benchmark_id']:
        raise ValueError('V2 protocol checksum differs')
    approved = read_json(Path(__file__).parent / 'atrnet_star_v2_approved_sources.json')
    oldhash = identity['source_hashes']
    expected = set(approved['runtime_names'])
    if set(oldhash) != expected:
        raise ValueError('V2 source fingerprint inventory is not the approved release')
    for name, h in oldhash.items():
        allowed = approved['suite_sha256'] if name == 'suite.json' else [approved['sha256'][name]]
        if h not in allowed:
            raise ValueError('V2 source is not allowlisted: ' + name)
    groups = {'training_sources': ['cnn_core.py', 'dataset.py', 'atr_image.py', 'backend.py', 'common.py', 'frozen_dataset_anchor.json'], 'optimizer_sources': ['optimizers.py', 'legacy_algorithms.py', 'hga_optimizer.py', 'journal.py'], 'checkpoint_sources': ['checkpoints.py', 'final_evaluation.py']}
    for field, names in groups.items():
        if set(b[field]) != set(names):
            raise ValueError('V2 source group inventory differs')
        for name, h in b[field].items():
            if h != oldhash[name]:
                raise ValueError('V2 source groups disagree')
    if b['schema'] != 'SOC10_N4800_formal_v2' or b['population'] != 5:
        raise ValueError('Not an approved V2 experiment')
    for key in ['mode', 'dataset_id', 'max_fe', 'runs', 'seed', 'precision', 'weights_sha256', 'backbone', 'input_size', 'smoke_epochs', 'torch_threads', 'num_workers', 'HGA_T', 'environment', 'FE', 'selection', 'test_policy']:
        if b[key] != new[key]:
            raise ValueError('V2/V5 experiment mismatch: ' + key)
    if identity['interval'] != current['identity']['interval']:
        raise ValueError('V2 interval differs; pass its original --interval')
    if not set(identity['algorithms']) <= set(LEGACY + HGA) or not set(identity['algorithms']) <= set(current['identity']['algorithms']):
        raise ValueError('Imported algorithm missing from requested V5 suite')
    if not set(p['selected_runs']) <= set(current['selected_runs']):
        raise ValueError('Imported run IDs not assigned to this output')
    return p

def _copy_checkpoint(old, new, old_args, new_args, events):
    if not (old / 'best_model.pt').exists():
        return False
    from src.HGA.atrnet_star.checkpoints import load_best, save_best
    complete = [r for r in events if r['new_fe'] and r['result']['status'] == 'complete']
    best = selected(complete)
    if best is None:
        raise ValueError('Model present without a complete journal selection')
    shutil.copy2(old / 'best_model.pt', new / 'best_model.pt')
    if (old / 'best_model.json').exists():
        shutil.copy2(old / 'best_model.json', new / 'best_model.json')
    expected = selection_identity(old_args, best)
    allowed = [selection_identity(old_args, r) for r in complete]
    payload = load_best(new, expected, allowed)
    if payload is None:
        import torch
        payload = torch.load(new / 'best_model.pt', map_location='cpu', weights_only=True)
        found = [r for r in complete if selection_identity(old_args, r) == payload['selection']]
        if len(found) != 1:
            raise ValueError('Stale checkpoint does not match V2 journal')
        best = found[0]
    save_best(new, selection_identity(new_args, best), payload['state_dict'])
    from src.HGA.atrnet_star.shared_resources import SharedWeights, weight_identity
    SharedWeights(new_args).publish(weight_identity(new_args, new_args.run, best), payload['state_dict'], f'track:{new_args.algorithm}:{new_args.run}')
    return best == selected(complete)

def seed_cache(args, events):
    folder = Path(args.cache_root) / args.cache_id
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'training.sqlite', timeout=120)
    try:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT NOT NULL,checksum TEXT NOT NULL)')
        for rec in events:
            if not rec['new_fe'] or rec['result']['status'] != 'complete' or args.algorithm == 'ASHA':
                continue
            result = dict(rec['result'])
            key = digest([args.cache_id, args.run, rec['candidate_id'], result['training_seed']])
            old = db.execute('SELECT payload,checksum FROM results WHERE key=?', (key,)).fetchone()
            if old:
                obj = json.loads(old[0])
                if digest(obj) != old[1] or any((obj[k] != result[k] for k in ['config', 'training_seed', 'f1', 'loss', 'best_epoch'])):
                    raise ValueError('Conflicting candidate result during migration')
            else:
                db.execute('INSERT INTO results VALUES (?,?,?)', (key, json.dumps(result, allow_nan=False), digest(result)))
        db.commit()
    finally:
        db.close()

def migrate(sources, current, args):
    output = Path(args.output)
    manifest = output / 'migration.json'
    prior = read_json(manifest) if manifest.exists() else None
    sources = sorted({str(Path(p).resolve()) for p in sources})
    if str(output) in sources:
        raise ValueError('V2 source must differ from V5 output')
    if prior and prior['sources'] != sources:
        raise ValueError('Migration sources changed')
    completed = dict(prior['completed']) if prior else {}
    source_ids = {}
    occupied = set()
    with ExitStack() as stack:
        for path in sources:
            if not (Path(path) / 'protocol.json').is_file():
                raise ValueError('V2 results path lacks protocol.json: ' + path)
            stack.enter_context(FileLock(Path(path) / 'experiment.lock'))
        for path in sources:
            source = Path(path)
            p = check_protocol(source, current)
            if args.mode == 'atr' and read_json(source / 'hardware.json') != read_json(output / 'hardware.json'):
                raise ValueError('V2 hardware differs; this migration supports the same device runtime only')
            shadow = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='v2_readonly_')))
            shutil.copy2(source / 'protocol.json', shadow / 'protocol.json')
            for name in p['identity']['algorithms']:
                for run in p['selected_runs']:
                    old = source / 'runs' / name / f'run_{run:02d}'
                    rd = shadow / 'runs' / name / f'run_{run:02d}'
                    rd.mkdir(parents=True)
                    for suffix in ['', '-journal', '-wal', '-shm']:
                        file = old / ('journal.sqlite' + suffix)
                        if file.exists():
                            shutil.copy2(file, rd / file.name)
                    if not (rd / 'journal.sqlite').exists():
                        raise ValueError('Missing V2 journal')
                    recovered = Journal(rd / 'journal.sqlite', None, args.target, run, args.seed, digest([p['protocol_id'], name, run]))
                    recovered.close()
            b, tracks, _ = collect([shadow])
            source_ids[path] = p['protocol_id']
            if args.mode == 'atr':
                from atrnet_star_dataset import validate_manifest
                snap = validate_manifest(read_json(source / 'frozen_dataset_snapshot.json'), args.dataset_id)
                if snap != read_json(output / 'frozen_dataset_snapshot.json'):
                    raise ValueError('Frozen dataset snapshot differs')
            for (name, run), rows in tracks.items():
                key = f'{name}:{run}'
                if key in occupied:
                    raise ValueError('Duplicate V2 algorithm/run source')
                occupied.add(key)
                old = source / 'runs' / name / f'run_{run:02d}'
                events = read_events(shadow / 'runs' / name / f'run_{run:02d}' / 'journal.sqlite')
                signature = digest(events)
                record = dict(source=path, source_protocol=p['protocol_id'], source_events_sha256=signature, completed_fe=len(rows))
                rd = output / 'runs' / name / f'run_{run:02d}'
                if key in completed:
                    if completed[key] != record:
                        raise ValueError('V2 source changed after migration')
                    if not rd.exists():
                        raise ValueError('Migrated track missing')
                    continue
                receipt = rd / 'migration_receipt.json'
                if receipt.exists():
                    if read_json(receipt) != record:
                        raise ValueError('Existing migration receipt differs')
                else:
                    if rd.exists() and any(rd.iterdir()):
                        raise ValueError('Cannot import over an existing V5 track')
                    stage = rd.with_name(rd.name + '.importing')
                    shutil.rmtree(stage, ignore_errors=True)
                    stage.mkdir(parents=True)
                    new_args = SimpleNamespace(**{**vars(args), 'algorithm': name, 'run': run})
                    old_args = SimpleNamespace(protocol_id=p['protocol_id'], dataset_id=args.dataset_id, algorithm=name, run=run)
                    j = Journal(stage / 'journal.sqlite', None, args.target, run, args.seed, digest([args.protocol_id, name, run]))
                    try:
                        with j.db:
                            for i, rec in enumerate(events):
                                j.db.execute('INSERT INTO calls VALUES (?,?,?)', (i, json.dumps(rec, allow_nan=False), hash_json(rec)))
                            j.db.execute("UPDATE meta SET value=? WHERE key='committed_fe'", (str(len(rows)),))
                    finally:
                        j.close()
                    if rows:
                        atomic_json(stage / 'selected.json', selected(rows))
                    current_weights = _copy_checkpoint(old, stage, old_args, new_args, events) if args.mode == 'atr' else False
                    seed_cache(new_args, events)
                    for filename in ['weight_reconstruction.json', 'runtime.log']:
                        if (old / filename).exists():
                            shutil.copy2(old / filename, stage / filename)
                    test = validate_test_result(old)
                    if test:
                        if len(rows) != 500 or test['selection'] != selection_identity(old_args, selected(rows)):
                            raise ValueError('V2 test selection differs from journal')
                        if current_weights:
                            if test['weights_sha256'] != file_hash(old / 'best_model.pt'):
                                raise ValueError('V2 test/model hash mismatch')
                            for filename in test['output_hashes']:
                                shutil.copy2(old / filename, stage / filename)
                            test = copy.deepcopy(test)
                            test['selection'] = selection_identity(new_args, selected(rows))
                            test['weights_sha256'] = file_hash(stage / 'best_model.pt')
                            test.pop('checksum')
                            test['checksum'] = digest(test)
                            atomic_json(stage / 'test_metrics.json', test)
                    atomic_json(stage / 'migration_receipt.json', record)
                    if rd.exists():
                        rd.rmdir()
                    os.replace(stage, rd)
                completed[key] = record
                atomic_json(manifest, dict(schema='v2-to-v4-migration', sources=sources, source_protocols=source_ids, completed=completed, note='V2 committed records copied unchanged; new task fingerprints and rebinding are audited; original sources retained.'))
    return completed
