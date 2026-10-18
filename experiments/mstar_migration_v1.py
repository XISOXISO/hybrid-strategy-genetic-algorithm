from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import copy, json, shutil, sqlite3, tempfile
from pathlib import Path
from types import SimpleNamespace
from mstar_common import FileLock, atomic_json, digest, file_hash, read_json
from src.HGA.mstar.journal import Journal, Paused, read_events
BASE = Path(__file__).parent

def reference_hashes():
    hashes = read_json(BASE / 'mstar_v1_reference_manifest.json')
    for name, expected in hashes.items():
        if file_hash(BASE / ('mstar_v1_' + name)) != expected:
            raise ValueError('Frozen V1 reference checksum differs: ' + name)
    return hashes

def legacy_benchmark(new):
    old = copy.deepcopy(new)
    old['schema'] = 'MSTAR_original6_B500_v1'
    old.pop('performance_contract', None)
    hashes = reference_hashes()
    for field in ('training_sources', 'optimizer_sources', 'checkpoint_sources', 'engineering_sources'):
        old[field] = {name: hashes[name] for name in old[field] if name in hashes}
    old['engineering_sources'] = {n: hashes[n] for n in ['hga_reuse.py', 'shared_resources.py', 'resilient_scheduler.py']}
    return old

def replay_saved(path, protocol, name, run):
    from src.baselines.mstar.algorithm_factory import run_optimizer
    from src.HGA.mstar.hga_optimizer import ControlledHGA
    events = read_events(path)
    target = sum((e['new_fe'] for e in events))
    if not target:
        return
    with tempfile.TemporaryDirectory() as tmp:
        j = Journal(Path(tmp) / 'replay.sqlite', None, target, run, protocol['identity']['benchmark']['seed'], 'offline-verification')
        try:
            source = sqlite3.connect(str(path), timeout=30)
            with j.db:
                for row in source.execute('SELECT seq,payload,checksum FROM calls ORDER BY seq'):
                    j.db.execute('INSERT INTO calls VALUES (?,?,?)', row)
                j.db.execute("UPDATE meta SET value=? WHERE key='committed_fe'", (str(target),))
            source.close()
            j.saved = read_events(j.path)
            try:
                seed = protocol['identity']['benchmark']['seed'] + run
                budget = protocol['identity']['benchmark']['max_fe']
                if name.startswith('HGA_'):
                    ControlledHGA(name[4:], j, seed, budget).run()
                else:
                    run_optimizer(name, j, seed, budget)
            except Paused:
                pass
            if j.fes != target or j.seq != len(events):
                raise ValueError('Offline replay differs from stored calls')
        finally:
            j.close()

def import_v1(source, output, protocol, hardware):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError('V1 import requires a NEW V2 results directory')
    if (output / 'protocol.json').exists():
        receipt = read_json(output / 'v1_import.json') if (output / 'v1_import.json').exists() else None
        if not receipt or receipt['source'] != str(source):
            raise ValueError('V2 results already exist without this V1 import')
        return
    with FileLock(source / 'experiment.lock'):
        old = read_json(source / 'protocol.json')
        identity = old['identity']
        if old['protocol_id'] != digest(identity) or identity['benchmark_id'] != digest(identity['benchmark']):
            raise ValueError('V1 protocol checksum mismatch')
        if identity['source_hashes'] != reference_hashes():
            raise ValueError('Unknown V1 runtime code; import rejected')
        if identity['benchmark'] != legacy_benchmark(protocol['identity']['benchmark']):
            raise ValueError('V1 dataset/environment/scientific protocol differs from V2')
        if identity['algorithms'] != protocol['identity']['algorithms'] or old['selected_runs'] != protocol['selected_runs'] or identity['interval'] != protocol['identity']['interval']:
            raise ValueError('V1 algorithm subset/run IDs/interval differ; keep the same subset during import')
        if old['identity']['benchmark']['mode'] == 'mstar' and read_json(source / 'hardware.json') != hardware:
            raise ValueError('V1 GPU/runtime hardware differs; exact training reuse is not certified')
        from report_mstar import collect, selected
        from run_mstar import count_fe
        for path in source.glob('runs/*/*/journal.sqlite'):
            count_fe(path)
        collect([source])
        stage = output / 'v1_import_stage'
        marker = output / 'v1_import_pending.json'
        keys = [(n, r) for r in old['selected_runs'] for n in identity['algorithms']]
        if marker.exists():
            plan = read_json(marker)
            if plan['source_protocol_id'] != old['protocol_id'] or plan['target_protocol_id'] != protocol['protocol_id']:
                raise ValueError('Pending migration protocol differs')
        else:
            if (output / 'runs').exists() and any((output / 'runs').glob('*/*/journal.sqlite')):
                raise ValueError('V2 destination contains unexpected histories')
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir()
            for name, run in keys:
                src = source / 'runs' / name / f'run_{run:02d}'
                dst = stage / 'runs' / name / f'run_{run:02d}'
                dst.mkdir(parents=True)
                replay_saved(src / 'journal.sqlite', old, name, run)
                db = sqlite3.connect(str(src / 'journal.sqlite'), timeout=30)
                target = sqlite3.connect(dst / 'journal.sqlite')
                try:
                    db.backup(target)
                    target.execute("UPDATE meta SET value=? WHERE key='fingerprint'", (digest([protocol['protocol_id'], name, run]),))
                    target.commit()
                finally:
                    target.close()
                    db.close()
                rows = read_events(dst / 'journal.sqlite')
                best = selected(rows)
                if best:
                    atomic_json(dst / 'selected.json', best)
                if (src / 'independent_reuse_fallback.json').exists():
                    shutil.copy2(src / 'independent_reuse_fallback.json', dst / 'independent_reuse_fallback.json')
                if protocol['identity']['benchmark']['mode'] == 'mstar' and best:
                    from src.HGA.mstar.checkpoints import selection_identity, load_best, save_best
                    args = SimpleNamespace(protocol_id=old['protocol_id'], dataset_id=identity['benchmark']['dataset_id'], algorithm=name, run=run)
                    expected = selection_identity(args, best)
                    allowed = [selection_identity(args, e) for e in rows if e['result']['status'] == 'complete']
                    old_copy = stage / 'checkpoint_sources' / name / f'run_{run:02d}'
                    old_copy.mkdir(parents=True)
                    for filename in ('best_model.pt', 'best_model.json'):
                        if (src / filename).exists():
                            shutil.copy2(src / filename, old_copy / filename)
                    payload = load_best(old_copy, expected, allowed)
                    if payload:
                        args.protocol_id = protocol['protocol_id']
                        save_best(dst, selection_identity(args, best), payload['state_dict'])
            plan = {'source': str(source), 'source_protocol_id': old['protocol_id'], 'target_protocol_id': protocol['protocol_id'], 'keys': [[n, r] for n, r in keys], 'imported_fe': sum((sum((e['new_fe'] for e in read_events(stage / 'runs' / n / f'run_{r:02d}' / 'journal.sqlite'))) for n, r in keys)), 'note': 'Metric payloads preserved; fingerprints and weight selection protocol migrated after exact optimizer replay. Prefix/test reports regenerated.'}
            atomic_json(marker, plan)
        for n, r in keys:
            src = stage / 'runs' / n / f'run_{r:02d}'
            dst = output / 'runs' / n / f'run_{r:02d}'
            if src.exists():
                if dst.exists():
                    raise ValueError('Migration destination collision')
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
            if not (dst / 'journal.sqlite').exists():
                raise ValueError('Incomplete migration staging')
        atomic_json(output / 'v1_import.json', plan)
        atomic_json(output / 'protocol.json', protocol)
        marker.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
