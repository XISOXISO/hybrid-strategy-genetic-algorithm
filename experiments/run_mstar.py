from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, importlib.metadata, math, os, signal, sqlite3, sys, time, traceback, uuid
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from mstar_common import FileLock, atomic_json, digest, file_hash, parse_runs, read_json, runtime_hashes
from mstar_dashboard import Dashboard
from src.HGA.mstar.journal import Journal, Paused, read_events
from src.HGA.mstar.shared_resources import CacheBusy
from mstar_resilient_scheduler import TrackScheduler, POLICY
from mstar_candidate_engine import read_cached
from mstar_backend import MSTARBackend
from report_mstar import report, collect
from mstar_run_audit import AttemptAudit, ResourceSampler, audit_contract
BASE = Path(__file__).parent

def performance_contract():
    return {'ordered_candidate_parallel': False, 'static_preprocess': 'V1 exact FP32', 'prefetch': 'sequential CPU augmentation/noise and default-stream GPU transfer', 'validation': 'epoch buffered, V1 Python summation order', 'training_equivalence': 'frozen V1 real-device one-epoch certificate required'}

def execution_contract():
    return {'version': 'V4', 'scheduling': 'spawned algorithm/run tracks', 'candidate_order': 'one candidate at a time per track', 'cache_contention': 'release process and requeue track', 'memory_cleanup': 'release unused CUDA allocator blocks after each training'}

def release_hashes():
    return {**runtime_hashes(), **{n: file_hash(BASE / ('mstar_' + n)) for n in ('run_audit.py', 'checkpoint_archive.py', 'paper_outputs.py', 'dashboard.py')}}

def configure_track_cache(a):
    if getattr(a, 'measurement_mode', 'shared') == 'isolated':
        a.cache_id = digest(['V4 isolated', a.cache_id, str(Path(a.output).resolve()), a.algorithm, a.run])
        a.legacy_cache_id = ''

class TrackBackend(MSTARBackend):

    def evaluate(self, genes, cid, trial=None):
        with self.audit.phase('evaluate', candidate_id=cid):
            result = self._evaluate_candidate(genes, cid, trial)
            self.audit.emit('candidate', candidate_id=cid, cache_hit=result.get('cache_hit', False), physical_training_seconds=result['physical_training_seconds'])
            return result

    def _evaluate_candidate(self, genes, cid, trial=None):
        self._candidate_state = None
        if trial is None and (not getattr(self.args, 'independent_reuse', False)):
            cached = read_cached(vars(self.args), self.run, cid)
            if cached is not None:
                self.status('cached', active_cid=cid)
                return cached
        return super().evaluate(genes, cid, trial)

    def train(self, cfg, seed, trial):
        with self.audit.phase('train', candidate_id=cfg['candidate_id']):
            result = super().train(cfg, seed, trial)
        from mstar_training_prefetch import release_candidate_memory
        release_candidate_memory(self.device, self.run_dir / 'gpu_memory.json', cfg['candidate_id'])
        return result

    def ensure_checkpoint(self, best):
        with self.audit.phase('checkpoint'):
            return super().ensure_checkpoint(best)

    def archive(self, fe):
        from mstar_checkpoint_archive import snapshot
        rows = list(self.journal.seen.values()) or [r for r in self.journal.saved if r['new_fe']]
        with self.audit.phase('archive'):
            snapshot(self.args, self.run_dir, rows, fe)

    def committed(self, fe, best):
        super().committed(fe, best)
        if fe % 50 == 0 or fe == self.args.max_fe:
            self.archive(fe)

def directory(output, algorithm, run):
    return Path(output) / 'runs' / algorithm / f'run_{run:02d}'

def task_id(protocol, algorithm, run):
    return digest([protocol, algorithm, run])

def count_fe(path):
    if not Path(path).exists():
        return 0
    db = sqlite3.connect(str(Path(path).resolve()), timeout=30)
    try:
        row = db.execute("SELECT value FROM meta WHERE key='committed_fe'").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc):
            return 0
        raise
    finally:
        db.close()

def initialize_worker(mode, torch_threads):
    if mode == 'mstar':
        import torch
        torch.set_num_interop_threads(1)
        torch.set_num_threads(torch_threads)

def worker(task):
    from mstar_backend import MSTARBackend, SyntheticBackend
    from src.HGA.mstar.hga_optimizer import ControlledHGA
    from src.baselines.mstar.algorithm_factory import run_optimizer
    a = SimpleNamespace(**task)
    rd = directory(a.output, a.algorithm, a.run)
    rd.mkdir(parents=True, exist_ok=True)
    with FileLock(rd / 'run.lock'), (rd / 'runtime.log').open('a', buffering=1, encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            atomic_json(rd / 'status.json', {'state': 'replaying', 'completed_fe': 0})
            a.independent_reuse = (rd / 'independent_reuse_fallback.json').exists()
            configure_track_cache(a)
            audit = AttemptAudit(rd, a, count_fe(rd / 'journal.sqlite'))
            b = None
            j = None
            controller = None
            start = time.monotonic()
            outcome = 'failed'
            try:
                b = (TrackBackend if a.mode == 'mstar' else SyntheticBackend)(a, a.run, rd)
                b.audit = audit
                j = Journal(rd / 'journal.sqlite', b, a.target, a.run, a.seed, task_id(a.protocol_id, a.algorithm, a.run))
                b.journal = j
                if a.mode == 'mstar' and j.saved:
                    from report_mstar import selected
                    b.ensure_checkpoint(selected([r for r in j.saved if r['new_fe']]))
                    b.archive(len([r for r in j.saved if r['new_fe']]))
                try:
                    if a.algorithm.startswith('HGA_'):
                        controller = ControlledHGA(a.algorithm[4:], j, a.seed + a.run, a.max_fe, population_size=10, max_generations=400, stagnation_generations=10)
                        if not a.disable_prefix and (not a.independent_reuse):
                            from src.HGA.mstar.hga_reuse import attach
                            attach(controller, j, a)
                        controller.run()
                    else:
                        controller = run_optimizer(a.algorithm, j, a.seed + a.run, a.max_fe)
                except Paused:
                    pass
                if j.fes != a.target:
                    raise RuntimeError(f'{a.algorithm} stopped at FE{j.fes}, expected {a.target}; no random fill')
                if a.mode == 'mstar':
                    b.ensure_checkpoint(j.best)
                if controller:
                    atomic_json(rd / 'diagnostics.json', controller.diagnostics())
                if j.best:
                    atomic_json(rd / 'selected.json', j.best)
                atomic_json(rd / 'done.json', {'completed_fe': j.fes, 'target': a.target, 'new_trainings': j.new_trainings, 'seconds_this_launch': time.monotonic() - start})
                atomic_json(rd / 'status.json', {'state': 'target_reached', 'completed_fe': j.fes, 'best_val_macro_f1': j.best['result']['f1'] if j.best else None})
                outcome = 'target_reached'
                return j.fes
            except CacheBusy as exc:
                atomic_json(rd / 'status.json', {'state': 'yielded_cache', 'completed_fe': count_fe(j.path), 'waiting_for': exc.path})
                outcome = 'yielded_cache'
                return {'blocked': exc.path, 'completed_fe': count_fe(j.path)}
            except BaseException as exc:
                traceback.print_exc()
                atomic_json(rd / 'status.json', {'state': 'error', 'completed_fe': count_fe(rd / 'journal.sqlite'), 'error': str(exc)})
                raise
            finally:
                try:
                    if j:
                        j.close()
                    if b:
                        b.close()
                finally:
                    audit.end(outcome, count_fe(rd / 'journal.sqlite'))

def audited_test_worker(task):
    from mstar_final_evaluation import test_worker
    a = SimpleNamespace(**task)
    configure_track_cache(a)
    rd = directory(a.output, a.algorithm, a.run)
    audit = AttemptAudit(rd, a, count_fe(rd / 'journal.sqlite'), stage='test')
    outcome = 'failed'
    try:
        with audit.phase('final_test'):
            result = test_worker(vars(a))
        outcome = 'test_complete'
        return result
    finally:
        audit.end(outcome, count_fe(rd / 'journal.sqlite'))

def env(mode):
    names = ['numpy', 'scipy']
    if mode == 'mstar':
        names += ['torch', 'torchvision', 'Pillow']
    return {'python': sys.version.split()[0], **{n: importlib.metadata.version(n) for n in names}}

def launch(a, ui):
    suite = read_json(BASE / 'mstar_suite.json')
    names = list(suite['algorithms'])
    if a.algorithms:
        names = a.algorithms.split(',')
        if len(names) != len(set(names)) or not set(names) <= set(suite['algorithms']):
            raise ValueError('Unsupported MSTAR algorithm')
    runs = parse_runs(a.run_ids, a.runs)
    target = a.stop_after or a.max_fe
    if not 1 <= target <= a.max_fe <= 6561 or min(a.jobs, a.torch_threads, a.interval, a.runs) < 1 or a.num_workers < 0 or (a.smoke_epochs < 0) or (a.max_retries < 0) or (min(a.retry_delay, a.stall_timeout, a.cache_wait_timeout) < 0) or (not all((math.isfinite(v) for v in (a.retry_delay, a.stall_timeout, a.cache_wait_timeout)))):
        raise ValueError('Invalid FE/run/job/thread/interval setting')
    if not math.isfinite(a.resource_interval) or a.resource_interval < 0.1:
        raise ValueError('Resource sampling interval must be >=0.1 seconds')
    if a.measurement_mode == 'isolated' and (a.jobs != 1 or a.v1_results):
        raise ValueError('Isolated measurement requires JOBS=1 and a fresh output without imports')
    if a.mode == 'mstar' and (a.runs != 10 or a.max_fe != 500 or a.torch_threads != 2 or (a.num_workers != 0)):
        raise ValueError('MSTAR protocol fixes 10 global runs, MAXFE500, torch-threads2 and num-workers0')
    a.output = str(Path(a.output).resolve())
    a.data = str(Path(a.data).resolve())
    a.cache_root = str(Path(a.cache_root).resolve())
    a.performance_cache = str(Path(a.performance_cache).resolve())
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / 'experiment.lock'):
        if (output / 'upgrade_import_pending.json').exists():
            raise ValueError('Incomplete upgrade import: repeat bash resume_existing.sh before training')
        (output / 'run_summary.json').unlink(missing_ok=True)
        dataset = 'synthetic-no-SAR'
        weights = 'none'
        hardware = {}
        if a.mode == 'mstar':
            from mstar_backend import preflight
            initialize_worker(a.mode, a.torch_threads)
            dataset, weights, hardware = preflight(a, output, ui)
            from mstar_dataset import load_prepared
            from mstar_tensor_cache import build_cache
            _, samples = load_prepared(a.data, expected_dataset_id=dataset)
            a.tensor_cache = build_cache(samples['train'] + samples['validation'], a.performance_cache, dataset, progress=lambda f: ui.update(10 + f, 'Build exact preprocessing cache', below=[f'Train/validation only: {f:.0%}']))
            from mstar_equivalence import certify
            certify(SimpleNamespace(**vars(a), dataset_id=dataset, environment=env(a.mode)), hardware, samples, ui)
        hashes = release_hashes()
        benchmark = {'schema': 'MSTAR_original6_B500_v2', 'mode': a.mode, 'dataset_id': dataset, 'max_fe': a.max_fe, 'runs': a.runs, 'seed': a.seed, 'precision': 'BF16-FP32 loss, deterministic algorithms' if a.mode == 'mstar' else 'synthetic', 'weights_sha256': weights, 'backbone': 'ResNet18-10class', 'input_size': 224, 'smoke_epochs': a.smoke_epochs, 'torch_threads': a.torch_threads, 'num_workers': a.num_workers, 'population': {'hga': 10, 'JSO': 147, 'LSHADE': 144}, 'DE_settings': {n: __import__('src.baselines.mstar.de_optimizers', fromlist=['*']).parameters(n, 8, a.max_fe) for n in ['JSO', 'LSHADE']}, 'HGA_T': 10, 'environment': env(a.mode), 'training_sources': {n: hashes[n] for n in ['cnn_core.py', 'dataset.py', 'backend.py', 'common.py', 'frozen_dataset_anchor.json']}, 'optimizer_sources': {n: hashes[n] for n in ['optimizers.py', 'hga_optimizer.py', 'de_optimizers.py', 'journal.py']}, 'FE': 'one distinct encoded config per track; nearest-unseen discrete repair; cached result still consumes one logical FE', 'selection': 'max validation macro-F1, then min validation loss; only complete trials eligible', 'test_evaluated': False, 'candidate_training_repeats': 1, 'optimizer_seed_rule': '42 + run', 'candidate_seed_rule': 'sha256(candidate,42,run,cid) mod (2**31-1)', 'preprocessing': 'PIL resize224 -> tensor -> crop224/flip/rotate5 -> ImageNet normalization', 'augmentation_strength_active': False, 'fixed_training_noise': {'std': 0.01, 'clamp': [-2.1179, 2.64], 'on_normalized_CPU_tensor': True}, 'smoke_only': bool(a.smoke_epochs), 'HGA_engine': 'controlled four variants from previously supplied four-variant engine, population10; not original multi-subpopulation class', 'test_policy': __import__('mstar_final_evaluation').TEST_POLICY, 'checkpoint_sources': {n: hashes[n] for n in ['checkpoints.py', 'final_evaluation.py']}, 'engineering_sources': {n: hashes[n] for n in ['hga_reuse.py', 'shared_resources.py', 'resilient_scheduler.py']}}
        benchmark['performance_contract'] = performance_contract()
        benchmark_id = digest(benchmark)
        identity = {'benchmark': benchmark, 'benchmark_id': benchmark_id, 'suite': suite['suite'], 'algorithms': names, 'interval': a.interval, 'source_hashes': hashes, 'execution': execution_contract(), 'audit': audit_contract(a.resource_interval, a.measurement_mode)}
        protocol_id = digest(identity)
        protocol = {'identity': identity, 'protocol_id': protocol_id, 'selected_runs': runs}
        pp = output / 'protocol.json'
        if pp.exists() and read_json(pp) != protocol:
            raise ValueError('Protocol/data/code/algorithm/run IDs changed; use a NEW output')
        if a.v1_results:
            from mstar_migration_v1 import import_v1
            ui.update(12, 'Validate / import V1 journals', force=True)
            import_v1(a.v1_results, output, protocol, hardware)
        elif (output / 'v1_import_pending.json').exists():
            raise ValueError('Incomplete V1 import: repeat command with --v1-results')
        atomic_json(pp, protocol)
        atomic_json(output / 'hardware.json', hardware)
        cache_id = digest([benchmark_id, hardware])
        receipt_path = output / 'upgrade_import.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            checked = dict(receipt)
            checksum = checked.pop('checksum', None)
            if checksum != digest(checked) or receipt['target_protocol_id'] != protocol_id:
                raise ValueError('Upgrade import receipt checksum/protocol mismatch')
            if receipt['source_hardware'] != hardware:
                raise ValueError('Imported cache belongs to a different GPU/runtime hardware identity')
            cache_id = receipt['source_cache_id']
            if not isinstance(cache_id, str) or len(cache_id) != 64 or any((c not in '0123456789abcdef' for c in cache_id)):
                raise ValueError('Invalid imported cache namespace')
        from mstar_migration_v1 import legacy_benchmark
        legacy_cache_id = digest([digest(legacy_benchmark(benchmark)), hardware])
        base = {**vars(a), 'target': target, 'dataset_id': dataset, 'protocol_id': protocol_id, 'cache_id': cache_id, 'legacy_cache_id': legacy_cache_id, 'selected_runs': runs, 'launch_id': uuid.uuid4().hex}
        keys = [(n, r) for r in runs for n in names]
        pending = []
        for n, r in keys:
            rd = directory(output, n, r)
            rd.mkdir(parents=True, exist_ok=True)
            j = Journal(rd / 'journal.sqlite', None, target, r, a.seed, task_id(protocol_id, n, r))
            try:
                events = j.saved
            finally:
                j.close()
            count = sum((e['new_fe'] for e in events))
            if count > a.max_fe:
                raise RuntimeError('Journal exceeds MAXFE')
            if count > target:
                raise ValueError('stop-after is below already committed FE; increase target, do not rewind results')
            if count != count_fe(rd / 'journal.sqlite'):
                raise RuntimeError('Journal progress metadata mismatch')
            done = read_json(rd / 'done.json') if (rd / 'done.json').exists() else {}
            if a.mode == 'mstar' or count < target or done.get('completed_fe', 0) < target:
                pending.append((n, r))
        collect([output])
        from mstar_migration_v1 import replay_saved
        for n, r in keys:
            replay_saved(directory(output, n, r) / 'journal.sqlite', protocol, n, r)
        interrupted = False
        failure = None
        marker = -1
        search_outcome = {'failed': []}
        report_failure = None
        last_counts = {key: count_fe(directory(output, *key) / 'journal.sqlite') for key in keys}

        def safe_report(plots=False):
            nonlocal report_failure
            try:
                with (output / 'reporting.log').open('a', encoding='utf-8') as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        result = report([output], output / 'reports', target, plots=plots)
                report_failure = None
                (output / 'report_error.json').unlink(missing_ok=True)
                return result
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                report_failure = str(exc)
                with (output / 'reporting.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
                atomic_json(output / 'report_error.json', {'error': str(exc), 'reports_current': False})
                return None

        def safe_paper(plots=False):
            try:
                from mstar_paper_outputs import generate
                generate([output], output / 'paper_outputs', plots=plots)
                (output / 'paper_error.json').unlink(missing_ok=True)
                return True
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                with (output / 'reporting.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
                atomic_json(output / 'paper_error.json', {'error': str(exc), 'reports_current': False})
                return False
        ui.update(12, 'Schedule independent tracks', above=[f"{suite['suite']} | runs {runs} | jobs={a.jobs}", f'{len(keys)} tracks x {target} UNIQUE FE; JSO / LSHADE only'], force=True)

        def recover_track(key):
            n, r = key
            j = Journal(directory(output, n, r) / 'journal.sqlite', None, target, r, a.seed, task_id(protocol_id, n, r))
            j.close()

        def search_display(snapshot):
            nonlocal marker
            sampler.update(scheduler)
            counts = {}
            for key in keys:
                try:
                    counts[key] = count_fe(directory(output, *key) / 'journal.sqlite')
                    last_counts[key] = counts[key]
                except sqlite3.DatabaseError:
                    counts[key] = last_counts[key]
            total = target * len(keys)
            completed = sum((min(c, target) for c in counts.values()))
            active = []
            fraction = 0.0
            for n, r in snapshot['active_tracks']:
                status_path = directory(output, n, r) / 'status.json'
                status = read_json(status_path) if status_path.exists() else {}
                line = f'{n} run {r}: committed {counts[n, r]}/{target}'
                if status.get('state') == 'training':
                    line += f" | CID{status.get('active_cid')} FE{status.get('active_fe')} {status.get('phase', 'training')} ep{status.get('epoch', 0)}/{status.get('epochs', 0)}"
                    fraction += status.get('active_fe_fraction', 0.0)
                else:
                    line += ' ' + status.get('state', 'starting')
                active.append(line)
            common = min(counts.values())
            atomic_json(output / 'live_progress.json', {'completed_fe': completed, 'target_fe': total, 'common_fe': common, 'active': active})
            ui.update(12 + 76 * min(total, completed + fraction) / total, 'Training / validation', above=[f'Committed FE {completed}/{total} | common FE{common}', f'jobs={a.jobs} | JSO / LSHADE only'], below=active + [f"Queued tracks {len(snapshot['queued_tracks'])} | cache waiters {len(snapshot['cache_waiters'])} | retry waiters {len(snapshot['retry_waiters'])} | failed tracks {len(snapshot['failed_tracks'])}", str(output)])
            cp = common // a.interval
            if cp > marker:
                safe_report(plots=False)
                marker = cp
        scheduler = TrackScheduler(output, a.jobs, base, max_retries=a.max_retries, retry_delay=a.retry_delay, stall_timeout=a.stall_timeout, cache_wait_timeout=a.cache_wait_timeout, initializer=initialize_worker, initargs=(a.mode, a.torch_threads))
        sampler = ResourceSampler(output, base['launch_id'], a.resource_interval, gpu=a.mode == 'mstar')
        try:
            if pending:
                search_outcome = scheduler.run(pending, worker, search_display, recover_track)
        except KeyboardInterrupt:
            interrupted = True
        except BaseException as exc:
            failure = exc
        finally:
            sampler.close()
            ui.update(ui.percent if interrupted or failure or search_outcome['failed'] else 89, 'Recover journals / save reports', force=True)
            for n, r in keys:
                try:
                    recover_track((n, r))
                except Exception as exc:
                    if not any((f['algorithm'] == n and f['run'] == r for f in search_outcome['failed'])):
                        search_outcome['failed'].append({'algorithm': n, 'run': r, 'stage': 'journal_recovery', 'error': str(exc)})
            summary = safe_report(plots=True)
            safe_paper(plots=False)
        if failure:
            raise failure
        if interrupted:
            return 130
        if search_outcome['failed'] or summary is None or (not summary['all_reached_target']):
            atomic_json(output / 'run_summary.json', {'state': 'INCOMPLETE', 'search': search_outcome, 'test_started': False, 'report_error': report_failure, 'note': 'Healthy tracks finished; failed tracks retained. Resume same command after inspecting recovery.log.'})
            ui.update(ui.percent, 'INCOMPLETE', above=[f"{len(search_outcome['failed'])} search tracks failed; healthy tracks finished", 'Test pending until all local selections reach FE500'], below=[str(output / 'recovery.log'), str(output / 'reports')], force=True)
            return 2
        tested = False
        if a.mode == 'mstar' and target == a.max_fe and (a.smoke_epochs == 0):
            from mstar_final_evaluation import test_worker
            ui.update(90, 'Independent test', above=['All local MAXFE500 selections frozen', '779 test images; no feedback to search'], force=True)

            def test_display(snapshot):
                test_sampler.update(tester)
                finished = len(snapshot['succeeded_tracks'])
                ui.update(90 + 8 * finished / len(keys), 'Independent test', above=[f'Tested models {finished}/{len(keys)}', f"Test retry wait {len(snapshot['retry_waiters'])} | failed {len(snapshot['failed_tracks'])}"], below=[str(output)])
            tester = TrackScheduler(output, a.jobs, base, stage='test', max_retries=a.max_retries, retry_delay=a.retry_delay, stall_timeout=a.stall_timeout, cache_wait_timeout=a.cache_wait_timeout, initializer=initialize_worker, initargs=(a.mode, a.torch_threads))
            test_sampler = ResourceSampler(output, base['launch_id'], a.resource_interval, gpu=True)
            try:
                test_outcome = tester.run(keys, audited_test_worker, test_display, recover_track)
            finally:
                test_sampler.close()
            if test_outcome['failed']:
                safe_report(plots=True)
                atomic_json(output / 'run_summary.json', {'state': 'INCOMPLETE', 'search': search_outcome, 'test': test_outcome})
                ui.update(ui.percent, 'INCOMPLETE', above=[f"{len(test_outcome['failed'])} test tracks failed; healthy tests finished"], below=[str(output / 'recovery.log'), str(output / 'reports')], force=True)
                return 2
            summary = safe_report(plots=True)
            if summary is None:
                atomic_json(output / 'run_summary.json', {'state': 'INCOMPLETE', 'report_error': report_failure, 'search': search_outcome, 'test': test_outcome})
                return 2
            tested = summary['test_complete_for_present_tracks']
            if not tested:
                raise RuntimeError('Independent test incomplete')
        if not safe_paper(plots=True):
            atomic_json(output / 'run_summary.json', {'state': 'INCOMPLETE', 'search': search_outcome, 'test_complete': tested, 'paper_error': read_json(output / 'paper_error.json')})
            return 2
        atomic_json(output / 'run_summary.json', {'state': 'COMPLETED', 'search': search_outcome, 'test_complete': tested})
        ui.update(100, 'Completed', above=[f'All {len(keys)} tracks reached FE{target} / MAXFE{a.max_fe}', 'Independent test completed.' if tested else 'Test pending until MAXFE500 (synthetic mode never tests SAR).'], below=[str(output / 'reports'), 'Resume: same protocol/output/run IDs; jobs may change.'], force=True)
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['mstar', 'synthetic'], default='mstar')
    p.add_argument('--data', default=str(_experiment_dir.parent / 'datasets/mstar/prepared'))
    p.add_argument('--root', default=str(_experiment_dir.parent / 'datasets/mstar/prepared'))
    p.add_argument('--cache-root', default=str(_experiment_dir.parent / '.experiment_cache/mstar/candidates'))
    p.add_argument('--performance-cache', default=str(_experiment_dir.parent / '.experiment_cache/mstar/performance'))
    p.add_argument('--v1-results', help='Validated migration from STOPPED V1 output; target must be new')
    p.add_argument('--output', default=str(_experiment_dir.parent / 'results/mstar'))
    p.add_argument('--algorithms')
    p.add_argument('--max-fe', type=int, default=500)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--run-ids', default='0-9')
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--interval', type=int, default=50)
    p.add_argument('--resource-interval', type=float, default=2.0)
    p.add_argument('--measurement-mode', choices=['shared', 'isolated'], default='shared')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--smoke-epochs', type=int, default=0)
    p.add_argument('--max-retries', type=int, default=3, help='Additional retries per task per launch; integrity failures never retry')
    p.add_argument('--retry-delay', type=float, default=10.0, help='Initial exponential retry delay in seconds (cap300)')
    p.add_argument('--stall-timeout', type=float, default=1800.0, help='Restart task after no status/journal changes; 0 disables')
    p.add_argument('--cache-wait-timeout', type=float, default=3600.0, help='Candidate lease wait limit; 0 disables')
    p.add_argument('--disable-prefix', action='store_true', help='Diagnostic: disable HGA state fork; metric/weight cache remains active')
    a = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def request_stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, request_stop)
    for n in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[n] = str(a.torch_threads)
    ui = Dashboard()
    try:
        code = launch(a, ui)
    except KeyboardInterrupt:
        code = 130
    except BaseException as exc:
        ui.update(ui.percent, 'ERROR', below=[str(exc)], force=True)
        raise
    finally:
        ui.close()
    raise SystemExit(code)
if __name__ == '__main__':
    main()
