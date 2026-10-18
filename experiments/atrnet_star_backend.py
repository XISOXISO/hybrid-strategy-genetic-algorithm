from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib, math, random, sqlite3, time
import numpy as np
from atrnet_star_common import atomic_json, decode_candidate, digest, file_hash, read_json, seed_for

def objective(f1, loss):
    return 1 - f1 + 1e-06 * min(loss, 10.0)

class SyntheticBackend:

    def __init__(self, args, run, run_dir=None, journal=None):
        self.args = args
        self.run = run
        self.calls = 0
        self.journal = journal

    def evaluate(self, genes, cid, trial=None):
        self.calls += 1
        np.random.seed(cid)
        random.seed(cid)
        seed = seed_for('candidate', self.args.seed, self.run, cid)
        cfg = decode_candidate(cid)
        quality = 0.96 - 0.013 * np.sum((np.asarray(genes) - [2, 1, 0, 2, 0, 1, 2, 1]) ** 2)
        reports = []
        pruned = False
        for ep in range(1, cfg['epochs'] + 1):
            f1 = float(quality - 0.08 / (ep + 1) + 0.0001 * self.run)
            loss = 1 - f1
            if trial is not None:
                val = objective(f1, loss)
                trial.report(val, ep)
                pruned = bool(trial.should_prune())
                reports.append([ep, val, pruned])
                if pruned:
                    break
        return dict(f1=f1, loss=loss, objective=objective(f1, loss), status='pruned' if pruned else 'complete', reports=reports, training_seconds=0.0, physical_training_seconds=0.0, epochs_completed=ep, cache_hit=False, numerical_failure=False, config=cfg, training_seed=seed, best_epoch=ep, peak_vram_mb=0.0, failure_reason='', epoch_history=[], effective_epochs=cfg['epochs'])

    def committed(self, fe, best):
        pass

    def close(self):
        pass

@contextmanager
def cache_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def preflight(args, output, ui):
    import torch
    import atrnet_star_cnn_core as core
    from atrnet_star_dataset import load_prepared
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 required')
    anchor = read_json(Path(__file__).parent / 'atrnet_star_frozen_dataset_anchor.json')
    data, _ = load_prepared(args.data, args.root, verify_train_val=True, expected_dataset_id=anchor['dataset_id'], image_source='copied', progress=lambda f, t: ui.update(10 * f, 'Verify frozen train/validation', below=[t]))
    with (output / 'initialization.log').open('a', encoding='utf-8') as log:
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(log), redirect_stderr(log):
            model = core.build_resnet18('imagenet')
            del model
    wp = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    wh = file_hash(wp)
    if not wh.startswith('f37072fd'):
        raise RuntimeError('Pretrained weights hash mismatch')
    atomic_json(output / 'frozen_dataset_snapshot.json', data)
    atomic_json(output / 'data_audit_snapshot.json', read_json(Path(args.data) / 'audit.json'))
    return (data['dataset_id'], wh, {'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)), 'runtime_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()})

class ATRBackend:

    def __init__(self, args, run, run_dir=None, journal=None):
        self.args = args
        self.run = run
        self.run_dir = Path(run_dir)
        self.journal = journal
        self.ready = False
        self.db = None
        self._candidate_state = None
        self._saved_identity = None
        from src.HGA.atrnet_star.shared_resources import SharedWeights
        self.shared_weights = SharedWeights(args)

    def status(self, state, **extra):
        j = self.journal
        atomic_json(self.run_dir / 'status.json', {'state': state, 'run': self.run, 'algorithm': self.args.algorithm, 'completed_fe': j.fes if j else 0, 'best_val_macro_f1': j.best['result']['f1'] if j and j.best else None, **extra})

    def prepare(self):
        if self.ready:
            return
        import torch
        import atrnet_star_cnn_core as core
        from atrnet_star_dataset import load_prepared
        self.torch = torch
        self.core = core
        self.device = torch.device('cuda:0')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA/BF16 unavailable')
        core.configure_performance(SimpleNamespace(performance_mode='safe', prefetch_factor=2, val_cache='auto', val_gpu_fraction=0.18, ram_image_cache='on', clear_cuda_cache_every=0))
        torch.use_deterministic_algorithms(True)
        self.status('loading')
        _, self.samples = load_prepared(self.args.data, verify_train_val=True, expected_dataset_id=self.args.dataset_id)
        core.GLOBAL_RAM_IMAGE_CACHE = core.preload_raw_image_cache(self.samples['train'] + self.samples['validation'], None)
        core.GLOBAL_VAL_STORE = core.EvalTensorStore(self.samples['validation'], None, self.device, 'auto')
        pretrained = torch.hub.load_state_dict_from_url(core.models.ResNet18_Weights.DEFAULT.url, progress=False, map_location='cpu', check_hash=True)

        def build(weights_mode):
            if weights_mode != 'imagenet':
                raise ValueError('Formal protocol requires ImageNet initialization')
            model = core.models.resnet18(weights=None)
            model.load_state_dict(pretrained)
            model.fc = core.nn.Linear(model.fc.in_features, 10)
            return model
        core.build_resnet18 = build
        self.cache_dir = Path(self.args.cache_root) / self.args.cache_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cache_dir / 'training.sqlite', timeout=120)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT NOT NULL,checksum TEXT NOT NULL)')
        self.db.commit()
        self.ready = True

    def disable_sharing(self, stage, exc):
        if not self.args.algorithm.startswith('HGA_'):
            raise exc
        self.args.independent_reuse = True
        atomic_json(self.run_dir / 'independent_reuse_fallback.json', dict(stage=stage, reason=str(exc), action='Keep checked own records; continue without candidate/prefix/weight sharing.'))

    def publish_shared(self, best, state):
        if getattr(self.args, 'independent_reuse', False):
            return
        from src.HGA.atrnet_star.shared_resources import weight_identity
        try:
            self.shared_weights.publish(weight_identity(self.args, self.run, best), state, f'track:{self.args.algorithm}:{self.run}')
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            self.disable_sharing('shared weights', exc)

    def evaluate(self, genes, cid, trial=None):
        self._candidate_state = None
        self.prepare()
        cfg = decode_candidate(cid)
        seed = seed_for('candidate', self.args.seed, self.run, cid)
        key = digest([self.args.cache_id, self.run, cid, seed])
        self.status('waiting_cache', active_cid=cid)
        if trial is not None or getattr(self.args, 'independent_reuse', False):
            return self.train(cfg, seed, trial)
        from src.HGA.atrnet_star.shared_resources import candidate_lock
        with candidate_lock(self.cache_dir / 'locks' / f'run_{self.run:02d}' / f'cid_{cid:04d}.lock'):
            got = self.db.execute('SELECT payload,checksum FROM results WHERE key=?', (key,)).fetchone()
            if got:
                import json
                try:
                    result = json.loads(got[0])
                    if digest(result) != got[1] or result['config'] != cfg or result['training_seed'] != seed or (result['status'] != 'complete'):
                        raise RuntimeError('Candidate cache checksum/config/seed mismatch')
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    self.disable_sharing('candidate cache', exc)
                    return self.train(cfg, seed, None)
                result['cache_hit'] = True
                result['physical_training_seconds'] = 0.0
                return result
            result = self.train(cfg, seed, None)
            if result['status'] == 'complete':
                if self.journal and (self.journal.best is None or (result['f1'], -result['loss']) > (self.journal.best['result']['f1'], -self.journal.best['result']['loss'])):
                    from src.HGA.atrnet_star.shared_resources import weight_identity
                    rec = {'candidate_id': cid, 'result': result}
                    self.publish_shared(rec, self._candidate_state[2])
                import json
                with self.db:
                    self.db.execute('INSERT INTO results VALUES (?,?,?)', (key, json.dumps(result, allow_nan=False), digest(result)))
            return result

    def train(self, cfg, seed, trial):
        training = dict(cfg)
        if self.args.smoke_epochs:
            training['epochs'] = self.args.smoke_epochs
        reports = []

        def pruning(ep, f1, loss):
            val = objective(f1, loss)
            trial.report(val, ep)
            prune = bool(trial.should_prune())
            reports.append([ep, val, prune])
            return prune

        def progress(phase, epoch, epochs, batch, batches, score):
            part = 1.0 if phase == 'epoch_done' else 0.8 if phase == 'validation' else 0.8 * batch / max(1, batches)
            self.status('training', active_fe=self.journal.fes + 1, active_cid=cfg['candidate_id'], phase=phase, epoch=epoch, epochs=epochs, batch=batch, batches=batches, active_fe_fraction=min(0.999, (epoch - 1 + part) / epochs), epoch_val_macro_f1=score)
        result = self.core.train_candidate(config=training, train_samples=self.samples['train'], val_samples=self.samples['validation'], device=self.device, num_workers=self.args.num_workers, seed=seed, weights_mode='imagenet', cache_dir=None, collect_train_metrics=False, return_best_state=True, progress_callback=progress, epoch_prune_callback=pruning if trial is not None else None)
        self._candidate_state = (cfg['candidate_id'], seed, result['best_state'])
        del result['model']
        fail = bool(result['numerical_failure'])
        f1 = 0.0 if fail else float(result['best_val_macro_f1'])
        loss = 10.0 if fail else float(result['best_val_loss'])
        status = 'failed' if fail else 'pruned' if result['pruned'] else 'complete'
        return dict(f1=f1, loss=loss, objective=objective(f1, loss), status=status, reports=reports, training_seconds=result['training_seconds'], physical_training_seconds=result['training_seconds'], epochs_completed=result['epochs_completed'], cache_hit=False, numerical_failure=fail, config=cfg, training_seed=seed, best_epoch=result['best_epoch'], peak_vram_mb=result['peak_vram_mb'], failure_reason=result['numerical_failure_reason'], epoch_history=result['epoch_history'], effective_epochs=training['epochs'])

    def ensure_checkpoint(self, best):
        if best is None:
            return
        from src.HGA.atrnet_star.checkpoints import selection_identity, load_best, save_best
        from atrnet_star_common import read_json
        import numpy as np, random, time
        expected = selection_identity(self.args, best)
        if self._saved_identity == expected:
            return
        records = list(getattr(self.journal, 'seen', {}).values()) + list(getattr(self.journal, 'saved', []))
        allowed = [selection_identity(self.args, r) for r in records if r['result']['status'] == 'complete']
        payload = load_best(self.run_dir, expected, allowed)
        if payload is not None:
            self._saved_identity = expected
            return
        candidate = self._candidate_state
        if candidate and candidate[:2] == (best['candidate_id'], best['result']['training_seed']):
            state = candidate[2]
        else:
            self.prepare()
            from src.HGA.atrnet_star.shared_resources import weight_identity, candidate_lock
            shared = weight_identity(self.args, self.run, best)
            lease = f'track:{self.args.algorithm}:{self.run}'
            state = None
            if not getattr(self.args, 'independent_reuse', False):
                try:
                    state = self.shared_weights.get(shared, lease)
                except (ValueError, RuntimeError, OSError, KeyError) as exc:
                    self.disable_sharing('shared weights', exc)
            if state is None:
                lock = self.cache_dir / 'locks' / f'run_{self.run:02d}' / f"cid_{best['candidate_id']:04d}.lock"
                from contextlib import nullcontext
                with nullcontext() if getattr(self.args, 'independent_reuse', False) else candidate_lock(lock):
                    if not getattr(self.args, 'independent_reuse', False):
                        try:
                            state = self.shared_weights.get(shared, lease)
                        except (ValueError, RuntimeError, OSError, KeyError) as exc:
                            self.disable_sharing('shared weights', exc)
                    if state is None:
                        np_state, py_state = (np.random.get_state(), random.getstate())
                        start = time.monotonic()
                        try:
                            recovered = self.train(best['result']['config'], best['result']['training_seed'], None)
                            original = best['result']
                            if recovered['status'] != 'complete' or any((recovered[k] != original[k] for k in ['f1', 'loss', 'best_epoch', 'training_seed', 'config'])):
                                raise RuntimeError('Best-weight reconstruction differs from committed validation result')
                            state = self._candidate_state[2]
                            self.publish_shared(best, state)
                        finally:
                            np.random.set_state(np_state)
                            random.setstate(py_state)
                        path = self.run_dir / 'weight_reconstruction.json'
                        old = read_json(path) if path.exists() else {'count': 0, 'seconds': 0.0}
                        atomic_json(path, {'count': old['count'] + 1, 'seconds': old['seconds'] + time.monotonic() - start, 'last_selection': expected, 'note': 'Missing-best recovery; not additional FE'})
            else:
                path = self.run_dir / 'weight_sharing.json'
                old = read_json(path) if path.exists() else {'count': 0}
                atomic_json(path, {'count': old['count'] + 1, 'last_shared_candidate': best['candidate_id']})
        self.publish_shared(best, state)
        save_best(self.run_dir, expected, state)
        self._saved_identity = expected

    def committed(self, fe, best):
        try:
            self.ensure_checkpoint(best)
        finally:
            self._candidate_state = None
        self.status('running')

    def close(self):
        self._candidate_state = None
        if self.db:
            self.db.close()
