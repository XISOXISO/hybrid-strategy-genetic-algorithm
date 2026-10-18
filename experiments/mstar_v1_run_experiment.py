from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, importlib.metadata, math, os, signal, sqlite3, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from mstar_v1_common import FileLock, atomic_json, digest, file_hash, parse_runs, read_json, runtime_hashes
from mstar_dashboard import Dashboard
from mstar_v1_journal import Journal, Paused, read_events
from mstar_v1_shared_resources import CacheBusy
from mstar_v1_resilient_scheduler import TrackScheduler, POLICY
from mstar_v1_reporting import report, collect
BASE = Path(__file__).parent

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
    from mstar_v1_backend import MSTARBackend, SyntheticBackend
    from mstar_v1_hga_optimizer import ControlledHGA
    from mstar_v1_optimizers import run_optimizer
    a = SimpleNamespace(**task)
    rd = directory(a.output, a.algorithm, a.run)
    rd.mkdir(parents=True, exist_ok=True)
    with FileLock(rd / 'run.lock'), (rd / 'runtime.log').open('a', buffering=1, encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            atomic_json(rd / 'status.json', {'state': 'replaying', 'completed_fe': 0})
            a.independent_reuse = (rd / 'independent_reuse_fallback.json').exists()
            b = (MSTARBackend if a.mode == 'mstar' else SyntheticBackend)(a, a.run, rd)
            j = Journal(rd / 'journal.sqlite', b, a.target, a.run, a.seed, task_id(a.protocol_id, a.algorithm, a.run))
            b.journal = j
            controller = None
            start = time.monotonic()
            try:
                if a.mode == 'mstar' and j.saved:
                    from mstar_v1_reporting import selected
                    b.ensure_checkpoint(selected([r for r in j.saved if r['new_fe']]))
                try:
                    if a.algorithm.startswith('HGA_'):
                        controller = ControlledHGA(a.algorithm[4:], j, a.seed + a.run, a.max_fe, population_size=10, max_generations=400, stagnation_generations=10)
                        if not a.disable_prefix and (not a.independent_reuse):
                            from mstar_v1_hga_reuse import attach
                            attach(controller, j, a)
                        controller.run()
                    else:
                        controller = run_optimizer(a.algorithm, j, a.seed + a.run, a.max_fe, 10)
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
                return j.fes
            except CacheBusy as exc:
                atomic_json(rd / 'status.json', {'state': 'yielded_cache', 'completed_fe': count_fe(j.path), 'waiting_for': exc.path})
                return {'blocked': exc.path, 'completed_fe': count_fe(j.path)}
            except BaseException as exc:
                traceback.print_exc()
                atomic_json(rd / 'status.json', {'state': 'error', 'completed_fe': j.fes, 'error': str(exc)})
                raise
            finally:
                j.close()
                b.close()

def env(mode):
    names = ['numpy', 'scipy']
    if mode == 'mstar':
        names += ['torch', 'torchvision', 'Pillow']
    return {'python': sys.version.split()[0], **{n: importlib.metadata.version(n) for n in names}}

def launch(a, ui):
    suite = read_json(BASE / 'mstar_v1_suite.json')
    names = suite['algorithms']
    if a.algorithms:
        names = a.algorithms.split(',')
        if len(names) != len(set(names)) or not set(names) <= set(suite['algorithms']):
            raise ValueError('Invalid algorithm subset')
    runs = parse_runs(a.run_ids, a.runs)
    target = a.stop_after or a.max_fe
    if not 1 <= target <= a.max_fe <= 6561 or min(a.jobs, a.torch_threads, a.interval, a.runs) < 1 or a.num_workers < 0 or (a.smoke_epochs < 0) or (a.max_retries < 0) or (min(a.retry_delay, a.stall_timeout, a.cache_wait_timeout) < 0) or (not all((math.isfinite(v) for v in (a.retry_delay, a.stall_timeout, a.cache_wait_timeout)))):
        raise ValueError('Invalid FE/run/job/thread/interval setting')
    if a.mode == 'mstar' and (a.runs != 10 or a.max_fe != 500 or a.torch_threads != 2 or (a.num_workers != 0)):
        raise ValueError('MSTAR protocol fixes 10 global runs, MAXFE500, torch-threads2 and num-workers0')
    a.output = str(Path(a.output).resolve())
    a.data = str(Path(a.data).resolve())
    a.cache_root = str(Path(a.cache_root).resolve())
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / 'experiment.lock'):
        (output / 'run_summary.json').unlink(missing_ok=True)
        dataset = 'synthetic-no-SAR'
        weights = 'none'
        hardware = {}
        if a.mode == 'mstar':
            from mstar_v1_backend import preflight
            dataset, weights, hardware = preflight(a, output, ui)
        hashes = runtime_hashes()
        benchmark = {'schema': 'MSTAR_original6_B500_v1', 'mode': a.mode, 'dataset_id': dataset, 'max_fe': a.max_fe, 'runs': a.runs, 'seed': a.seed, 'precision': 'BF16-FP32 loss, deterministic algorithms' if a.mode == 'mstar' else 'synthetic', 'weights_sha256': weights, 'backbone': 'ResNet18-10class', 'input_size': 224, 'smoke_epochs': a.smoke_epochs, 'torch_threads': a.torch_threads, 'num_workers': a.num_workers, 'population': {'hga': 10, 'JSO': 147, 'LSHADE': 144}, 'DE_settings': {n: __import__('mstar_v1_de_optimizers').parameters(n, 8, a.max_fe) for n in ['JSO', 'LSHADE']}, 'HGA_T': 10, 'environment': env(a.mode), 'training_sources': {n: hashes[n] for n in ['cnn_core.py', 'dataset.py', 'backend.py', 'common.py', 'frozen_dataset_anchor.json']}, 'optimizer_sources': {n: hashes[n] for n in ['optimizers.py', 'hga_optimizer.py', 'de_optimizers.py', 'journal.py']}, 'FE': 'one distinct encoded config per track; nearest-unseen discrete repair; cached result still consumes one logical FE', 'selection': 'max validation macro-F1, then min validation loss; only complete trials eligible', 'test_evaluated': False, 'candidate_training_repeats': 1, 'optimizer_seed_rule': '42 + run', 'candidate_seed_rule': 'sha256(candidate,42,run,cid) mod (2**31-1)', 'preprocessing': 'PIL resize224 -> tensor -> crop224/flip/rotate5 -> ImageNet normalization', 'augmentation_strength_active': False, 'fixed_training_noise': {'std': 0.01, 'clamp': [-2.1179, 2.64], 'on_normalized_CPU_tensor': True}, 'smoke_only': bool(a.smoke_epochs), 'HGA_engine': 'controlled four variants from previously supplied four-variant engine, population10; not original multi-subpopulation class', 'test_policy': __import__('mstar_v1_final_evaluation').TEST_POLICY, 'checkpoint_sources': {n: hashes[n] for n in ['checkpoints.py', 'final_evaluation.py']}, 'engineering_sources': {n: hashes[n] for n in ['hga_reuse.py', 'shared_resources.py', 'resilient_scheduler.py']}}
        benchmark_id = digest(benchmark)
        identity = {'benchmark': benchmark, 'benchmark_id': benchmark_id, 'suite': suite['suite'], 'algorithms': names, 'interval': a.interval, 'source_hashes': hashes}
        protocol_id = digest(identity)
        protocol = {'identity': identity, 'protocol_id': protocol_id, 'selected_runs': runs}
        pp = output / 'protocol.json'
        if pp.exists() and read_json(pp) != protocol:
            raise ValueError('Protocol/data/code/algorithm/run IDs changed; use a NEW output')
        atomic_json(pp, protocol)
        atomic_json(output / 'hardware.json', hardware)
        cache_id = digest([benchmark_id, hardware])
        base = {**vars(a), 'target': target, 'dataset_id': dataset, 'protocol_id': protocol_id, 'cache_id': cache_id}
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
            if count != count_fe(rd / 'journal.sqlite'):
                raise RuntimeError('Journal progress metadata mismatch')
            done = read_json(rd / 'done.json') if (rd / 'done.json').exists() else {}
            if a.mode == 'mstar' or count < target or done.get('completed_fe', 0) < target:
                pending.append((n, r))
        collect([output])
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
        ui.update(12, 'Schedule independent tracks', above=[f"{suite['suite']} | runs {runs} | jobs={a.jobs}", f'{len(keys)} tracks x {target} UNIQUE FE; JSO/LSHADE -> RM -> CM -> SW60 -> SW50'], force=True)

        def recover_track(key):
            n, r = key
            j = Journal(directory(output, n, r) / 'journal.sqlite', None, target, r, a.seed, task_id(protocol_id, n, r))
            j.close()

        def search_display(snapshot):
            nonlocal marker
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
                rd = directory(output, n, r)
                status = read_json(rd / 'status.json') if (rd / 'status.json').exists() else {}
                best = status.get('best_val_macro_f1')
                best_text = f'{best:.5f}' if best is not None else '--'
                line = f"{n} run {r}: FE{counts[n, r]}/{target} best F1={best_text} {status.get('state', 'starting')}"
                if status.get('state') == 'training':
                    line += f" ep{status.get('epoch', 0)}/{status.get('epochs', 0)}"
                    if status.get('completed_fe') == counts[n, r]:
                        fraction += status.get('active_fe_fraction', 0.0)
                active.append(line)
            common = min(counts.values())
            atomic_json(output / 'live_progress.json', {'completed_fe': completed, 'target_fe': total, 'common_fe': common, 'active': active})
            ui.update(12 + 76 * min(total, completed + fraction) / total, 'Training / validation', above=[f'Committed FE {completed}/{total} | common FE{common}', f'jobs={a.jobs} | JSO/LSHADE -> RM -> CM -> SW60 -> SW50'], below=active + [f"Queued {len(snapshot['queued_tracks'])} | cache wait {len(snapshot['cache_waiters'])} | retry wait {len(snapshot['retry_waiters'])} | failed {len(snapshot['failed_tracks'])}", str(output)])
            cp = common // a.interval
            if cp > marker:
                safe_report(plots=False)
                marker = cp
        scheduler = TrackScheduler(output, a.jobs, base, max_retries=a.max_retries, retry_delay=a.retry_delay, stall_timeout=a.stall_timeout, cache_wait_timeout=a.cache_wait_timeout, initializer=initialize_worker, initargs=(a.mode, a.torch_threads))
        try:
            if pending:
                search_outcome = scheduler.run(pending, worker, search_display, recover_track)
        except KeyboardInterrupt:
            interrupted = True
        except BaseException as exc:
            failure = exc
        finally:
            ui.update(ui.percent if interrupted or failure or search_outcome['failed'] else 89, 'Recover journals / save reports', force=True)
            for n, r in keys:
                try:
                    recover_track((n, r))
                except Exception as exc:
                    if not any((f['algorithm'] == n and f['run'] == r for f in search_outcome['failed'])):
                        search_outcome['failed'].append({'algorithm': n, 'run': r, 'stage': 'journal_recovery', 'error': str(exc)})
            summary = safe_report(plots=True)
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
            from mstar_v1_final_evaluation import test_worker
            ui.update(90, 'Independent test', above=['All local MAXFE500 selections frozen', '779 test images; no feedback to search'], force=True)

            def test_display(snapshot):
                finished = len(snapshot['succeeded_tracks'])
                ui.update(90 + 8 * finished / len(keys), 'Independent test', above=[f'Tested models {finished}/{len(keys)}', f"Test retry wait {len(snapshot['retry_waiters'])} | failed {len(snapshot['failed_tracks'])}"], below=[str(output)])
            tester = TrackScheduler(output, a.jobs, base, stage='test', max_retries=a.max_retries, retry_delay=a.retry_delay, stall_timeout=a.stall_timeout, cache_wait_timeout=a.cache_wait_timeout, initializer=initialize_worker, initargs=(a.mode, a.torch_threads))
            test_outcome = tester.run(keys, test_worker, test_display, recover_track)
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
        atomic_json(output / 'run_summary.json', {'state': 'COMPLETED', 'search': search_outcome, 'test_complete': tested})
        ui.update(100, 'Completed', above=[f'All {len(keys)} tracks reached FE{target} / MAXFE{a.max_fe}', 'Independent test completed.' if tested else 'Test pending until MAXFE500 (synthetic mode never tests SAR).'], below=[str(output / 'reports'), 'Resume: same protocol/output/run IDs; jobs may change.'], force=True)
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['mstar', 'synthetic'], default='mstar')
    p.add_argument('--data', default=str(_experiment_dir.parent / 'datasets/mstar/prepared'))
    p.add_argument('--root', default=str(_experiment_dir.parent / 'datasets/mstar/prepared'))
    p.add_argument('--cache-root', default=str(_experiment_dir.parent / '.experiment_cache/mstar/candidates'))
    p.add_argument('--output', default=str(_experiment_dir.parent / 'results/mstar'))
    p.add_argument('--algorithms')
    p.add_argument('--max-fe', type=int, default=500)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--run-ids', default='0-9')
    p.add_argument('--jobs', type=int, default=9)
    p.add_argument('--interval', type=int, default=50)
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
