from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse
import concurrent.futures as futures
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import random
import signal
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VERSION = 'cec17-r100-v1.0'
STOP_EVENT = None

class BudgetComplete(BaseException):
    pass

class RunPaused(BaseException):
    pass

class ReplayMismatch(RuntimeError):
    pass

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def atomic_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

@contextlib.contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()

def dim_for(fid):
    return 30 if fid <= 2 else 20 if fid <= 5 else 10

def integers(spec, low, high):
    result = []
    for token in spec.replace(',', ' ').split():
        if '-' in token:
            left, right = map(int, token.split('-', 1))
            if left > right:
                raise ValueError(f'Invalid range: {token}')
            result.extend(range(left, right + 1))
        else:
            result.append(int(token))
    if not result or any((i < low or i > high for i in result)):
        raise ValueError(f'Expected values in {low}..{high}')
    return sorted(set(result))

def algorithm_selection(tokens):
    from cec2022_algorithm_registry import ALGORITHMS, ALIASES
    names = {x.lower(): x for x in ALGORITHMS}
    names.update({k.lower(): v for k, v in ALIASES.items()})
    result = []
    for token in tokens:
        for name in token.split(','):
            if name.lower() not in names:
                raise ValueError(f'Unknown algorithm: {name}')
            if names[name.lower()] not in result:
                result.append(names[name.lower()])
    return result

def hardware(device):
    cpu = platform.processor()
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
        except OSError:
            pass
    elif os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    info = {'hostname': socket.gethostname(), 'os': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(), 'python': platform.python_version(), 'device': device}
    if hasattr(os, 'sysconf'):
        try:
            info['ram_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (OSError, ValueError):
            pass
    elif os.name == 'nt':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info['ram_bytes'] = status.total
    versions = {}
    for name in ('numpy', 'scipy', 'optuna', 'opfunu', 'matplotlib', 'threadpoolctl'):
        versions[name] = importlib.metadata.version(name)
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is unavailable')
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        info['gpu'] = properties.name
        info['gpu_memory_bytes'] = properties.total_memory
        info['cuda_runtime'] = torch.version.cuda
        versions['torch'] = torch.__version__
    info['packages'] = versions
    return info

def evaluator(fid, dim, device):
    import numpy as np
    from cec2022_cec_evaluator import import_cec_classes, make_cpu_function, TorchCEC2022
    reference = make_cpu_function(fid, dim, import_cec_classes())
    if device == 'cpu':
        reference.evaluate(np.zeros(dim, dtype=np.float64))
        reference.n_fe = 0
        return reference.evaluate
    import torch
    target = TorchCEC2022(fid, dim, reference, device)
    rng = np.random.default_rng(20261009 + fid * 100 + dim)
    for x in np.vstack([np.zeros(dim), rng.uniform(-100, 100, (8, dim))]):
        expected = reference.evaluate(x)
        observed = float(target.evaluate(x).item())
        if not np.isclose(expected, observed, rtol=1e-09, atol=1e-07):
            raise RuntimeError(f'CPU/GPU FP64 validation failed: F{fid}/D{dim}')
    torch.cuda.synchronize()
    return lambda x: float(target.evaluate(x).item())

def support_hashes():
    import numpy as np
    from cec2022_cec_evaluator import import_cec_classes, make_cpu_function
    classes = import_cec_classes()
    hashes = {}
    for fid in range(1, 13):
        f = make_cpu_function(fid, dim_for(fid), classes)
        h = hashlib.sha256()
        for key in ('f_shift', 'f_matrix', 'f_shuffle'):
            if hasattr(f, key):
                data = np.ascontiguousarray(getattr(f, key))
                h.update(key.encode())
                h.update(str(data.shape).encode())
                h.update(str(data.dtype).encode())
                h.update(data.tobytes())
        hashes[f'F{fid}'] = h.hexdigest()
    return hashes

def make_protocol(args):
    from cec2022_algorithm_registry import ALGORITHMS
    from cec2022_algorithm_registry import source_hashes
    hashes = source_hashes()
    core = {'version': VERSION, 'algorithms': ALGORITHMS, 'runs': args.runs, 'cases': [{'fid': f, 'dim': dim_for(f), 'dimension_label': 'D30_EXT' if f <= 2 else f'D{dim_for(f)}', 'budget': int(args.fe_per_dim * dim_for(f))} for f in range(1, 13)], 'base_seed': args.base_seed, 'seed_rule': 'base_seed + run_id', 'precision': 'float64', 'device': args.device, 'workers': args.workers, 'threads': args.threads, 'checkpoint_every': args.checkpoint_every, 'hardware': hardware(args.device), 'source_sha256': hashes, 'support_sha256': support_hashes(), 'D30_official': False, 'HGA_T': 10, 'HGA_RM': 'original Algorithm_2.py mutation formula', 'HGA_CM': 'same original mutation-rate multiset, reversed rank assignment', 'ASHA': 'unchanged legacy static CEC implementation and parameters', 'DE': 'source-aligned v4 Python mechanism ports; NumPy RNG; CEC2022 budget', 'TPE': 'legacy sampler/startup; total trial count capped at exact FE budget', 'AdamHPO': 'legacy finite-difference Adam plus deterministic partial budget tail', 'CMA_boundary': 'legacy reflection repeated periodically when one reflection remains outside bounds', 'timing': 'initialization + search + evaluations; checkpoint I/O and replay excluded', 'timing_eligibility': 'uninterrupted runs only; workers>1 means shared-load timing', 'resume': 'seeded deterministic replay; cached vectors verified bit-for-bit', 'trace': 'every FE, best-so-far raw objective and error from bias'}
    return {'core': core, 'protocol_hash': digest_json(core), 'created_utc': datetime.now(timezone.utc).isoformat()}

class RecordedObjective:

    def __init__(self, job, evaluate, checkpoint, checkpoint_every, stop_after=0):
        import numpy as np
        self.job, self.evaluate, self.checkpoint = (job, evaluate, checkpoint)
        self.budget, self.fe = (job['budget'], 0)
        self.best, self.best_x = (float('inf'), None)
        self.xs, self.values = ([], [])
        self.curve = []
        self.prior_elapsed = 0.0
        self.prior_eval_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.started, self.io_seconds = (None, 0.0)
        self.finished_elapsed = None
        self.pauses, self.resumes, self.replayed = (0, 0, 0)
        self.checkpoint_every, self.stop_after = (checkpoint_every, stop_after)
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                for key in ('protocol_hash', 'algorithm', 'fid', 'dim', 'budget', 'run', 'seed'):
                    if meta[key] != job[key]:
                        raise RuntimeError(f'Checkpoint mismatch: {key}')
                self.xs = [x.copy() for x in z['x']]
                self.values = z['values'].tolist()
                self.curve = np.minimum.accumulate(z['values']).tolist()
                self.prior_elapsed = meta['elapsed_seconds']
                self.prior_eval_seconds = meta.get('evaluation_seconds', 0.0)
                self.pauses = meta.get('pauses', 0)
                self.resumes = meta.get('resumes', 0) + 1
        self.prefix = len(self.values)
        if self.prefix > self.budget:
            raise RuntimeError('Checkpoint exceeds FE budget')
        if self.prefix == 0:
            self.started = time.perf_counter()

    def elapsed(self):
        if self.finished_elapsed is not None:
            return self.finished_elapsed
        active = 0.0 if self.started is None else time.perf_counter() - self.started - self.io_seconds
        return self.prior_elapsed + max(0.0, active)

    def metadata(self):
        return {**self.job, 'fe_used': len(self.values), 'elapsed_seconds': self.elapsed(), 'evaluation_seconds': self.prior_eval_seconds + self.evaluation_seconds, 'pauses': self.pauses, 'resumes': self.resumes, 'replayed_fe': self.replayed}

    def save(self):
        import numpy as np
        start = time.perf_counter()
        atomic_npz(self.checkpoint, x=np.asarray(self.xs, dtype=np.float64).reshape(-1, self.job['dim']), values=np.asarray(self.values, dtype=np.float64), metadata=np.asarray(json.dumps(self.metadata())))
        if self.started is not None:
            self.io_seconds += time.perf_counter() - start

    def __call__(self, x):
        import numpy as np
        if self.fe >= self.budget:
            raise BudgetComplete()
        if STOP_EVENT is not None and STOP_EVENT.is_set() or (self.stop_after and self.fe >= self.stop_after):
            raise RunPaused()
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape != (self.job['dim'],) or not np.all(np.isfinite(x)):
            raise ValueError('Nonfinite candidate or wrong dimension')
        if np.any(x < -100.0) or np.any(x > 100.0):
            raise ValueError('Candidate outside the common [-100,100] search bounds')
        if self.fe < self.prefix:
            if not np.array_equal(x, self.xs[self.fe]):
                raise ReplayMismatch(f'Candidate differs at FE {self.fe + 1}; checkpoint retained')
            value = self.values[self.fe]
            self.replayed += 1
            self.fe += 1
            if value < self.best:
                self.best, self.best_x = (value, x.copy())
            if self.fe == self.prefix:
                self.started = time.perf_counter()
            return value
        start = time.perf_counter()
        value = float(self.evaluate(x))
        self.evaluation_seconds += time.perf_counter() - start
        if not np.isfinite(value):
            raise ValueError('Objective returned nonfinite fitness')
        self.xs.append(x.copy())
        self.values.append(value)
        self.fe += 1
        if value < self.best:
            self.best, self.best_x = (value, x.copy())
        self.curve.append(self.best)
        if self.fe == self.budget:
            self.finished_elapsed = self.elapsed()
            raise BudgetComplete()
        if self.fe % self.checkpoint_every == 0:
            self.save()
        return value

def initialize_worker(stop_event, threads):
    global STOP_EVENT
    STOP_EVENT = stop_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)

def run_job(job, output, checkpoint_every=25, stop_after=0, remeasure=False):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from cec2022_algorithm_registry import create, configuration
    output = Path(output)
    relative = Path(job['algorithm']) / f"F{job['fid']}" / f"run_{job['run']:03d}"
    result_path = output / 'runs' / relative.with_suffix('.npz')
    checkpoint = output / 'checkpoints' / relative.with_suffix('.npz')
    failure = output / 'errors' / relative.with_suffix('.json')
    with file_lock(output / 'locks' / relative.with_suffix('.lock')) as locked:
        if not locked:
            return {**job, 'status': 'locked'}
        previous = None
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                if meta['protocol_hash'] != job['protocol_hash']:
                    raise RuntimeError('Completed result protocol mismatch')
                if not remeasure or meta['timing_eligible']:
                    return {**job, 'status': 'skipped'}
                previous = {'metadata': meta, 'values': z['objective_values'].copy()}
            if checkpoint.exists():
                checkpoint.unlink()
        with threadpool_limits(limits=job['threads']):
            obj = None
            try:
                evaluate = evaluator(job['fid'], job['dim'], job['device'])
                random.seed(job['seed'])
                np.random.seed(job['seed'])
                obj = RecordedObjective(job, evaluate, checkpoint, checkpoint_every, stop_after)
                if obj.prefix < obj.budget:
                    try:
                        optimizer = create(job['algorithm'], job['dim'], job['budget'], job['seed'], obj)
                        optimizer.optimize()
                    except BudgetComplete:
                        pass
                if len(obj.values) != obj.budget:
                    raise RuntimeError(f'Short budget: {len(obj.values)}/{obj.budget}')
                values = np.asarray(obj.values, dtype=np.float64)
                curve = np.minimum.accumulate(values)
                if previous is not None and (not np.array_equal(values, previous['values'])):
                    raise ReplayMismatch('Timing remeasurement changed the objective trajectory')
                best_index = int(np.argmin(values))
                meta = {**obj.metadata(), 'status': 'ok', 'best_value': float(curve[-1]), 'timing_eligible': obj.pauses == 0 and obj.resumes == 0, 'shared_load_timing': job['workers'] > 1, 'configuration': configuration(job['algorithm'], job['dim'], job['budget']), 'completed_utc': datetime.now(timezone.utc).isoformat()}
                if previous is not None:
                    meta['timing_remeasurements'] = previous['metadata'].get('timing_remeasurements', 0) + 1
                    atomic_json(output / 'timing_audit' / relative.with_suffix('.json'), {'previous': previous['metadata'], 'replacement': meta})
                atomic_npz(result_path, fe=np.arange(1, obj.budget + 1), objective_values=values, best_values=curve, best_x=np.asarray(obj.xs[best_index], dtype=np.float64), metadata=np.asarray(json.dumps(meta)))
                if checkpoint.exists():
                    checkpoint.unlink()
                if failure.exists():
                    failure.unlink()
                return meta
            except RunPaused:
                obj.pauses += 1
                obj.save()
                return {**obj.metadata(), 'status': 'paused'}
            except Exception as error:
                if obj is not None and (not isinstance(error, ReplayMismatch)):
                    obj.save()
                record = {**job, 'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()}
                atomic_json(failure, record)
                return record

def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '--output-dir', default=str(ROOT.parent / 'results' / 'cec2022'))
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--run-ids', default='')
    parser.add_argument('--functions', default='1-12')
    parser.add_argument('--algorithms', nargs='+', default=None)
    parser.add_argument('--exclude-algorithms', nargs='+', default=[])
    parser.add_argument('--list-algorithms', action='store_true')
    parser.add_argument('--fe-per-dim', type=int, default=100)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', '--jobs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--stop-after-fe', type=int, default=0)
    parser.add_argument('--remeasure-interrupted', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--mstar-summary', default='')
    args = parser.parse_args()
    for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[n] = str(args.threads)
    from cec2022_algorithm_registry import ALGORITHMS
    if args.list_algorithms:
        print('\n'.join(ALGORITHMS))
        return 0
    if min(args.runs, args.workers, args.threads, args.fe_per_dim, args.checkpoint_every) < 1:
        parser.error('runs/workers/threads/fe-per-dim/checkpoint-every must be positive')
    if args.base_seed < 0 or args.base_seed + args.runs >= 2 ** 32:
        parser.error('Seed range must fit uint32')
    if args.device != 'cpu' and (not args.device.startswith('cuda')):
        parser.error('device must be cpu or cuda[:index]')
    output = Path(args.output).resolve()
    from summarize_cec2022 import summarize
    if args.summarize_only:
        summarize(output, not args.no_plots, args.mstar_summary)
        return 0
    selected = algorithm_selection(args.algorithms or ALGORITHMS)
    excluded = algorithm_selection(args.exclude_algorithms)
    selected = [a for a in selected if a not in excluded]
    if not selected:
        parser.error('No algorithms selected')
    functions = integers(args.functions, 1, 12)
    run_ids = integers(args.run_ids, 0, args.runs - 1) if args.run_ids else list(range(args.runs))
    if args.fe_per_dim < 19 and any((a in selected for a in ('LSHADE', 'JSO'))):
        parser.error('DE author initial populations require at least 19D for these dimensions')
    with file_lock(output / 'protocol.lock') as locked:
        if not locked:
            raise RuntimeError('Another process is preparing the same output')
        protocol = make_protocol(args)
        path = output / 'protocol.json'
        if path.exists():
            existing = json.loads(path.read_text(encoding='utf-8'))
            if existing['protocol_hash'] != protocol['protocol_hash']:
                changed = [k for k in protocol['core'] if existing['core'].get(k) != protocol['core'][k]]
                raise RuntimeError('Output protocol differs (' + ', '.join(changed) + '); use the original parameters or a new output directory')
            protocol = existing
        else:
            atomic_json(path, protocol)
    jobs = []
    skipped = 0
    import numpy as np
    for fid in functions:
        for algorithm in selected:
            for run in run_ids:
                job = {'algorithm': algorithm, 'fid': fid, 'dim': dim_for(fid), 'budget': args.fe_per_dim * dim_for(fid), 'run': run, 'seed': args.base_seed + run, 'device': args.device, 'workers': args.workers, 'threads': args.threads, 'protocol_hash': protocol['protocol_hash']}
                path = output / 'runs' / algorithm / f'F{fid}' / f'run_{run:03d}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        meta = json.loads(str(z['metadata'].item()))
                    if meta['protocol_hash'] != protocol['protocol_hash']:
                        raise RuntimeError(f'Unexpected protocol in {path}')
                    if not args.remeasure_interrupted or meta['timing_eligible']:
                        skipped += 1
                        continue
                jobs.append(job)
    print(f'CEC2022 | selected={len(selected)} | runs={args.runs} | device={args.device} | workers={args.workers} | threads={args.threads}', flush=True)
    print(f'Already complete={skipped} | pending={len(jobs)} | output={output}', flush=True)
    context = mp.get_context('spawn')
    stop_event = context.Event()
    interrupts = [0]

    def stop(signum, frame):
        interrupts[0] += 1
        stop_event.set()
        print('Saving checkpoints at the next FE boundary...', flush=True)
        if interrupts[0] > 1:
            raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGINT, stop)
    counts = {'ok': 0, 'paused': 0, 'error': 0, 'locked': 0, 'skipped': 0}
    pending = {}
    iterator = iter(jobs)
    try:
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=initialize_worker, initargs=(stop_event, args.threads)) as pool:

            def submit_next():
                if stop_event.is_set():
                    return
                job = next(iterator, None)
                if job is not None:
                    future = pool.submit(run_job, job, str(output), args.checkpoint_every, args.stop_after_fe, args.remeasure_interrupted)
                    pending[future] = job
            for _ in range(args.workers):
                submit_next()
            last_progress = time.monotonic()
            while pending:
                done, _ = futures.wait(pending, timeout=1, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {'status': 'error', 'error': str(error)}
                    counts[result['status']] += 1
                    print(f"{sum(counts.values())}/{len(jobs)} | {job['algorithm']} F{job['fid']} run={job['run']} | {result['status']} FE={result.get('fe_used', 0)}/{job['budget']}", flush=True)
                    if result['status'] == 'error':
                        print(result['error'], flush=True)
                    if result['status'] == 'paused':
                        stop_event.set()
                    submit_next()
                if time.monotonic() - last_progress >= 30:
                    labels = [f"{j['algorithm']}/F{j['fid']}/run{j['run']}" for j in pending.values()]
                    print('Running: ' + ', '.join(labels), flush=True)
                    last_progress = time.monotonic()
    finally:
        signal.signal(signal.SIGINT, old_handler)
    with file_lock(output / 'summary.lock') as locked:
        if locked:
            summarize(output, not args.no_plots, args.mstar_summary)
    print('Saved. ' + json.dumps(counts), flush=True)
    return 2 if counts['error'] else 0

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--output', '--output-dir', default=str(ROOT.parent / 'results' / 'cec2022'))
    args, _ = pre.parse_known_args()
    with file_lock(Path(args.output).resolve() / 'execution.lock') as locked:
        if not locked:
            raise RuntimeError('This output is already running; wait for that process to finish')
        return execute()
if __name__ == '__main__':
    mp.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Stopped; the latest saved checkpoints are retained.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
