from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import copy, json, math, multiprocessing, os, sqlite3, threading, time, traceback
from concurrent.futures import Future, wait
from pathlib import Path
from types import SimpleNamespace
from mstar_common import FileLock, atomic_json, decode_candidate, digest, read_json, seed_for
from mstar_backend import MSTARBackend
from mstar_resilient_scheduler import priority, recoverable, describe_error, stop_process

class StopRequested(BaseException):
    pass

def validate_result(result, args, run, cid):
    cfg = decode_candidate(cid)
    epochs = (args.smoke_epochs if args.mode == 'mstar' else 0) or cfg['epochs']
    if result.get('config') != cfg or result.get('training_seed') != seed_for('candidate', args.seed, run, cid) or result.get('status') != 'complete' or (result.get('epochs_completed') != epochs) or (result.get('effective_epochs') != epochs) or (not 1 <= result.get('best_epoch', 0) <= epochs) or (not math.isfinite(result.get('loss', float('nan')))) or (result['loss'] < 0) or (not 0 <= result.get('f1', float('nan')) <= 1) or (result.get('objective') != 1 - result['f1']):
        raise ValueError('Candidate result configuration/seed/epochs/metrics mismatch')

def read_cached(base, run, cid):
    args = SimpleNamespace(**base)
    for namespace in dict.fromkeys([args.cache_id, getattr(args, 'legacy_cache_id', '')]):
        path = Path(args.cache_root) / namespace / 'training.sqlite'
        if not namespace or not path.exists():
            continue
        key = digest([namespace, run, cid, seed_for('candidate', args.seed, run, cid)])
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)
        try:
            row = db.execute('SELECT payload,checksum FROM results WHERE key=?', (key,)).fetchone()
        except sqlite3.OperationalError as exc:
            if 'no such table' in str(exc):
                row = None
            else:
                raise
        finally:
            db.close()
        if row:
            result = json.loads(row[0])
            if digest(result) != row[1]:
                raise ValueError('Candidate cache checksum mismatch')
            validate_result(result, args, run, cid)
            result['cache_hit'] = True
            result['physical_training_seconds'] = 0.0
            return result
    return None

def _training_process(connection, mode, threads):
    from run_mstar import initialize_worker
    initialize_worker(mode, threads)
    backend = None
    try:
        connection.send({'ready': True})
        while True:
            task = connection.recv()
            if task is None:
                break
            try:
                a = SimpleNamespace(**task)
                cid = a.cid
                if task.get('_crash_once'):
                    marker = Path(task['_crash_once'])
                    if not marker.exists():
                        marker.write_text('crashed')
                        os._exit(71)
                if task.get('_delay'):
                    time.sleep(task['_delay'])
                if mode == 'synthetic':
                    from mstar_backend import SyntheticBackend
                    result = SyntheticBackend(a, a.run).evaluate(decode_candidate(cid)['genes'], cid)
                else:
                    if backend is None:
                        backend = MSTARBackend(a, a.run, a.job_dir)
                        backend.prepare()
                    else:
                        backend.args = a
                        backend.run = a.run
                        backend.run_dir = Path(a.job_dir)
                        backend._candidate_state = None
                        backend._saved_identity = None
                    backend.journal = SimpleNamespace(fes=a.fe - 1, best=None)
                    cfg = decode_candidate(cid)
                    seed = seed_for('candidate', a.seed, a.run, cid)
                    if a.force:
                        result = backend.train(cfg, seed, None)
                        from src.HGA.mstar.checkpoints import save_best, selection_identity
                        rec = {'fe': a.fe, 'candidate_id': cid, 'result': result}
                        save_best(a.job_dir, selection_identity(a, rec), backend._candidate_state[2])
                    else:
                        result = backend.evaluate(cfg['genes'], cid)
                    backend._candidate_state = None
                validate_result(result, a, a.run, cid)
                connection.send({'result': result})
            except BaseException as exc:
                from src.HGA.mstar.shared_resources import CacheBusy
                if isinstance(exc, CacheBusy):
                    connection.send({'blocked': exc.path})
                else:
                    error = describe_error(exc)
                    print(error['traceback'], flush=True)
                    connection.send({'error': error})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if backend:
            backend.close()
        connection.close()

def training_process(connection, mode, threads, logfile, parent_pid):
    import ctypes, signal
    if os.name == 'posix':
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise RuntimeError('Could not install parent-death signal')
        if os.getppid() != parent_pid:
            return
    from contextlib import redirect_stdout, redirect_stderr
    with Path(logfile).open('a', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                _training_process(connection, mode, threads)
            except BaseException:
                traceback.print_exc()
                raise

class CandidateBroker:

    def __init__(self, output, jobs, base, max_retries=3, retry_delay=10.0, stall_timeout=1800.0, cache_wait_timeout=3600.0, **unused):
        self.output = Path(output)
        self.jobs = jobs
        self.base = base
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stall_timeout = stall_timeout
        self.cache_wait_timeout = cache_wait_timeout
        self.startup_failures = 0
        self.lock = threading.RLock()
        self.entries = {}
        self.slots = {}
        self.cursors = {}
        self.stop = threading.Event()
        self.drivers = []
        self.failures = []
        self.succeeded = []
        self.context = multiprocessing.get_context('spawn')
        self.counter = 0
        self.lookahead = max(2, math.ceil(jobs / max(1, len(set(base['selected_runs'])))))
        self.started = time.monotonic()
        self.peak = 0
        self.physical_jobs = 0
        self.events = []

    def submit(self, algorithm, run, cid, fe, force=False):
        with self.lock:
            token = (algorithm, run, fe)
            key = (run, cid)
            if force:
                self.counter += 1
                key = (run, cid, 'force', self.counter)
            if key not in self.entries:
                cached = None if force else read_cached(self.base, run, cid)
                future = Future()
                self.counter += 1
                lease = f'pending:{run}:{cid}' + (f':{self.counter}' if force else '')
                self.entries[key] = {'future': future, 'clients': set(), 'owner': token, 'algorithm': algorithm, 'run': run, 'cid': cid, 'fe': fe, 'force': force, 'number': self.counter, 'lease': lease, 'ready_at': 0.0, 'attempts': 0, 'state': 'queued', 'cached': cached is not None}
                if cached:
                    self.entries[key]['state'] = 'complete'
                    future.set_result(cached)
            entry = self.entries[key]
            entry['clients'].add(token)
            return entry

    def result(self, entry, token):
        future = entry['future']
        while not self.stop.is_set():
            done, _ = wait((future,), timeout=0.2)
            if future in done:
                result = future.result()
                break
        else:
            raise StopRequested()
        result = copy.deepcopy(result)
        if entry['cached'] or token != entry['owner']:
            result['cache_hit'] = True
            result['physical_training_seconds'] = 0.0
        return result

    def advance(self, algorithm, run, fe, cid=None):
        leases = []
        with self.lock:
            self.cursors[algorithm, run] = max(fe, self.cursors.get((algorithm, run), 0))
            for entry in self.entries.values():
                entry['clients'] = {t for t in entry['clients'] if not (t[0] == algorithm and t[1] == run and (t[2] <= fe))}
                if entry['state'] == 'complete' and (not entry['clients']) and (not entry.get('released')):
                    if not entry['force']:
                        leases.append(entry['lease'])
                    entry['released'] = True
        if self.base['mode'] == 'mstar':
            from src.HGA.mstar.shared_resources import SharedWeights
            store = SharedWeights(SimpleNamespace(**self.base))
            for lease in leases:
                try:
                    store.release(lease)
                except (ValueError, OSError, KeyError) as exc:
                    self.events.append({'action': 'weight_cleanup_deferred', 'error': str(exc)})
                    break

    def drop(self, key):
        self.advance(*key, 10 ** 9)

    def finish_force(self, entry):
        with self.lock:
            entry['clients'].clear()
        self.advance(entry['algorithm'], entry['run'], self.cursors.get((entry['algorithm'], entry['run']), 0))

    def new_slot(self, index):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=training_process, args=(child, self.base['mode'], self.base['torch_threads'], str(self.output / f'training_slot_{index}.log'), os.getpid()))
        process.start()
        child.close()
        self.slots[index] = {'process': process, 'pipe': parent, 'ready': False, 'entry': None, 'started': time.monotonic(), 'activity': time.monotonic(), 'stamp': None}

    def fail_entry(self, entry, error):
        retry = recoverable(error) and entry['attempts'] <= self.max_retries
        event = {'run': entry['run'], 'candidate_id': entry['cid'], 'attempt': entry['attempts'], 'action': 'retry_candidate' if retry else 'fail_candidate', 'error': error}
        self.events.append(event)
        if retry:
            entry['state'] = 'queued'
            entry['ready_at'] = time.monotonic() + min(300.0, self.retry_delay * 2 ** max(0, entry['attempts'] - 1))
        else:
            entry['state'] = 'failed'
            if not entry['future'].done():
                entry['future'].set_exception(RuntimeError(json.dumps(error)))

    def snapshot(self):
        with self.lock:
            active = []
            for slot in self.slots.values():
                entry = slot['entry']
                if entry:
                    status_path = Path(entry['job_dir']) / 'status.json'
                    status = read_json(status_path) if status_path.exists() else {}
                    active.append({k: entry[k] for k in ('algorithm', 'run', 'cid', 'fe')} | {'status': status})
            return {'stage': 'candidate_search', 'jobs': self.jobs, 'active_candidates': active, 'active_tracks': [list(k) for k, t in self.drivers if t.is_alive()], 'queued_tracks': [list(k) for k, t in self.drivers if t.is_alive() and self.cursors.get(k, 0) == 0], 'cache_waiters': [e['cid'] for e in self.entries.values() if e.get('blocked_since') and e['state'] == 'queued'], 'retry_waiters': [e['cid'] for e in self.entries.values() if e['state'] == 'queued' and e['ready_at'] > time.monotonic()], 'failed_tracks': self.failures, 'succeeded_tracks': [list(k) for k in self.succeeded], 'queued_candidates': sum((e['state'] == 'queued' and bool(e['clients']) for e in self.entries.values())), 'buffered_candidates': sum((e['state'] == 'complete' and bool(e['clients']) for e in self.entries.values())), 'peak_training_candidates': self.peak, 'physical_jobs_this_launch': self.physical_jobs, 'policy': 'JSO/LSHADE -> RM -> CM -> SW60 -> SW50; ordered FE commit; no common barrier'}

    def drive(self, key):
        n, r = key
        task = {**self.base, 'algorithm': n, 'run': r}
        rd = self.output / 'runs' / n / f'run_{r:02d}'
        from src.HGA.mstar.journal import Journal, Paused
        from report_mstar import selected
        from src.baselines.mstar.algorithm_factory import run_optimizer
        from src.HGA.mstar.hga_optimizer import ControlledHGA
        try:
            with FileLock(rd / 'run.lock'):
                task['independent_reuse'] = (rd / 'independent_reuse_fallback.json').exists()
                b = BrokerBackend(SimpleNamespace(**task), r, rd, self)
                j = Journal(rd / 'journal.sqlite', b, task['target'], r, task['seed'], digest([task['protocol_id'], n, r]))
                b.journal = j
                try:
                    with self.lock:
                        self.cursors[key] = sum((e['new_fe'] for e in j.saved))
                    if task['mode'] == 'mstar' and j.saved:
                        chosen = selected([e for e in j.saved if e['new_fe']])
                        b.ensure_checkpoint(chosen)
                        if chosen and (not b.args.independent_reuse):
                            from src.HGA.mstar.checkpoints import load_best, selection_identity
                            own = load_best(rd, selection_identity(b.args, chosen))
                            b.publish_shared(chosen, own['state_dict'])
                    controller = None
                    try:
                        if n.startswith('HGA_'):
                            controller = ControlledHGA(n[4:], j, task['seed'] + r, task['max_fe'])
                            if not task['disable_prefix'] and (not b.args.independent_reuse):
                                from src.HGA.mstar.hga_reuse import attach
                                attach(controller, j, b.args)
                                with self.lock:
                                    self.cursors[key] = max(self.cursors[key], j.fes)
                            controller.run()
                        else:
                            controller = run_optimizer(n, j, task['seed'] + r, task['max_fe'])
                    except Paused:
                        pass
                    if j.fes != task['target']:
                        raise RuntimeError('Optimizer stopped before target; no random filling')
                    if controller:
                        atomic_json(rd / 'diagnostics.json', controller.diagnostics())
                    if j.best:
                        atomic_json(rd / 'selected.json', j.best)
                    atomic_json(rd / 'done.json', {'completed_fe': j.fes, 'target': task['target'], 'new_trainings': j.new_trainings})
                    atomic_json(rd / 'status.json', {'state': 'target_reached', 'completed_fe': j.fes})
                    with self.lock:
                        self.succeeded.append(key)
                finally:
                    j.close()
                    b.close()
        except StopRequested:
            pass
        except BaseException as exc:
            with (rd / 'search_worker_errors.log').open('a') as log:
                log.write(traceback.format_exc())
            with self.lock:
                self.failures.append({'algorithm': n, 'run': r, 'error': str(exc)})
        finally:
            self.drop(key)

    def run(self, keys, function=None, on_update=None, recover=None):
        try:
            for key in keys:
                thread = threading.Thread(target=self.drive, args=(key,), daemon=True)
                self.drivers.append((key, thread))
                thread.start()
            while any((t.is_alive() for _, t in self.drivers)):
                now = time.monotonic()
                with self.lock:
                    for index, slot in list(self.slots.items()):
                        pipe = slot['pipe']
                        entry = slot['entry']
                        process = slot['process']
                        message = None
                        while not pipe.closed and pipe.poll():
                            try:
                                got = pipe.recv()
                            except EOFError:
                                break
                            if got.get('ready'):
                                slot['ready'] = True
                            else:
                                message = got
                        if message and entry:
                            slot['entry'] = None
                            if 'result' in message:
                                validate_result(message['result'], SimpleNamespace(**self.base), entry['run'], entry['cid'])
                                entry['state'] = 'complete'
                                self.physical_jobs += int(not message['result']['cache_hit'])
                                entry['future'].set_result(message['result'])
                            elif 'blocked' in message:
                                entry['attempts'] -= 1
                                entry.setdefault('blocked_since', now)
                                if self.cache_wait_timeout and now - entry['blocked_since'] > self.cache_wait_timeout:
                                    entry['state'] = 'failed'
                                    entry['future'].set_exception(RuntimeError('External candidate lease wait timed out; healthy tracks continue'))
                                    self.events.append({'run': entry['run'], 'candidate_id': entry['cid'], 'action': 'cache_wait_timeout'})
                                else:
                                    entry['state'] = 'queued'
                                    entry['ready_at'] = now + 0.5
                            else:
                                self.fail_entry(entry, message['error'])
                                if recoverable(message['error']):
                                    stop_process(process)
                            entry = None
                        if entry:
                            path = Path(entry['job_dir']) / 'status.json'
                            stamp = path.stat().st_mtime_ns if path.exists() else 0
                            if stamp != slot['stamp']:
                                slot['stamp'] = stamp
                                slot['activity'] = now
                        stale = self.stall_timeout and (now - slot['activity'] > self.stall_timeout if entry else not slot['ready'] and now - slot['started'] > max(300.0, self.stall_timeout))
                        if not process.is_alive() or stale:
                            if entry:
                                self.fail_entry(entry, {'chain': [{'type': 'ProcessExited' if not stale else 'StalledTask', 'message': f'Training process exited/stalled (exit {process.exitcode})', 'errno': None}], 'traceback': ''})
                            if entry is None and (not slot['ready']):
                                self.startup_failures += 1
                                if self.startup_failures > self.max_retries:
                                    for waiting in self.entries.values():
                                        if not waiting['future'].done():
                                            waiting['state'] = 'failed'
                                            waiting['future'].set_exception(RuntimeError('Training worker initialization repeatedly failed; inspect worker logs'))
                                    self.startup_failures = 0
                            stop_process(process)
                            pipe.close()
                            del self.slots[index]
                    queued = []
                    for e in self.entries.values():
                        if e['state'] != 'queued' or not e['clients'] or e['ready_at'] > now:
                            continue
                        if not e['force'] and (not any((t[2] <= self.cursors.get(t[:2], 0) + self.lookahead for t in e['clients']))):
                            continue
                        queued.append(e)
                    queued.sort(key=lambda e: (min((priority(t[:2])[0] for t in e['clients'])), e['fe'], e['run'], e['number']))
                    for e in queued:
                        idle = next((i for i, s in self.slots.items() if s['ready'] and s['entry'] is None), None)
                        if idle is None:
                            if len(self.slots) < self.jobs:
                                index = next((i for i in range(self.jobs) if i not in self.slots))
                                self.new_slot(index)
                            continue
                        e['attempts'] += 1
                        e['state'] = 'running'
                        e['job_dir'] = str(self.output / 'training_jobs' / f"r{e['run']}_c{e['cid']}_a{e['attempts']}_{e['number']}")
                        Path(e['job_dir']).mkdir(parents=True, exist_ok=True)
                        task = {**self.base, **{k: e[k] for k in ('algorithm', 'run', 'cid', 'fe', 'force', 'job_dir')}, 'pending_lease': e['lease']}
                        hooks = self.base.get('_test_hooks', {}).get(str(e['cid']), {})
                        task.update(hooks)
                        self.slots[idle]['entry'] = e
                        self.slots[idle]['activity'] = now
                        self.slots[idle]['stamp'] = None
                        try:
                            self.slots[idle]['pipe'].send(task)
                        except (BrokenPipeError, EOFError, OSError):
                            self.fail_entry(e, {'chain': [{'type': 'ProcessExited', 'message': 'Training process disconnected before receiving candidate', 'errno': None}], 'traceback': ''})
                            bad = self.slots.pop(idle)
                            stop_process(bad['process'])
                            bad['pipe'].close()
                            continue
                        with (self.output / 'candidate_dispatch.log').open('a') as log:
                            log.write(json.dumps({'algorithm': e['algorithm'], 'run': e['run'], 'candidate_id': e['cid'], 'fe': e['fe'], 'attempt': e['attempts'], 'slot': idle, 'monotonic': now}) + '\n')
                    self.peak = max(self.peak, sum((s['entry'] is not None for s in self.slots.values())))
                snapshot = self.snapshot()
                atomic_json(self.output / 'scheduler.json', snapshot)
                atomic_json(self.output / 'candidate_recovery.json', {'events': self.events})
                if on_update:
                    on_update(snapshot)
                time.sleep(0.1)
            return {'failed': self.failures, 'succeeded': [list(k) for k in self.succeeded], 'retry_events': len(self.events), 'peak_training_candidates': self.peak, 'physical_jobs': self.physical_jobs}
        finally:
            self.stop.set()
            for slot in self.slots.values():
                stop_process(slot['process'])
                slot['pipe'].close()
            for entry in self.entries.values():
                if not entry['future'].done():
                    entry['future'].set_exception(StopRequested())
            for _, thread in self.drivers:
                thread.join(timeout=15)
            if any((t.is_alive() for _, t in self.drivers)):
                raise RuntimeError('Optimizer driver did not stop; journals must remain locked')
            self.slots.clear()

class BrokerBackend(MSTARBackend):
    isolated_worker_rng = True

    def __init__(self, args, run, run_dir, broker):
        self.args = args
        self.run = run
        self.run_dir = Path(run_dir)
        self.broker = broker
        self.journal = None
        self.ready = False
        self.db = None
        self._candidate_state = None
        self._saved_identity = None
        self.requests = {}
        self.current = None
        if args.mode == 'mstar':
            from src.HGA.mstar.shared_resources import SharedWeights
            self.shared_weights = SharedWeights(args)

    def prepare(self):
        self.cache_dir = Path(self.args.cache_root) / self.args.cache_id
        self.ready = True

    def prefetch_many(self, plans):
        for plan in plans:
            token = (self.args.algorithm, self.run, plan['fe'])
            self.requests[plan['cid']] = self.broker.submit(*token[:2], plan['cid'], plan['fe'])

    def evaluate(self, genes, cid, trial=None):
        fe = self.journal.fes + 1
        token = (self.args.algorithm, self.run, fe)
        if getattr(self.args, 'independent_reuse', False) and self.args.mode == 'mstar':
            return self.train(decode_candidate(cid), seed_for('candidate', self.args.seed, self.run, cid), trial)
        try:
            entry = self.requests.get(cid) or self.broker.submit(self.args.algorithm, self.run, cid, fe)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            if not self.args.algorithm.startswith('HGA_'):
                raise
            self.disable_sharing('candidate cache', exc)
            return self.train(decode_candidate(cid), seed_for('candidate', self.args.seed, self.run, cid), trial)
        self.current = (entry, token)
        self.status('waiting_candidate', active_cid=cid)
        return self.broker.result(entry, token)

    def train(self, cfg, seed, trial=None):
        entry = self.broker.submit(self.args.algorithm, self.run, cfg['candidate_id'], self.journal.fes + 1, force=True)
        result = self.broker.result(entry, entry['owner'])
        from src.HGA.mstar.checkpoints import load_best, selection_identity
        record = {'fe': entry['fe'], 'candidate_id': cfg['candidate_id'], 'result': result}
        payload = load_best(entry['job_dir'], selection_identity(self.args, record))
        if payload is None:
            raise ValueError('Reconstructed candidate weight missing')
        state = payload['state_dict']
        (Path(entry['job_dir']) / 'best_model.pt').unlink(missing_ok=True)
        (Path(entry['job_dir']) / 'best_model.json').unlink(missing_ok=True)
        self._candidate_state = (cfg['candidate_id'], seed, state)
        self.broker.finish_force(entry)
        return result

    def ensure_checkpoint(self, best):
        from src.HGA.mstar.shared_resources import CacheBusy
        started = time.monotonic()
        waits = 0
        while not self.broker.stop.is_set():
            try:
                super().ensure_checkpoint(best)
                if waits:
                    atomic_json(self.run_dir / 'checkpoint_wait.json', {'state': 'resolved', 'candidate_id': best['candidate_id'], 'polls': waits, 'seconds': time.monotonic() - started})
                return
            except CacheBusy as exc:
                waits += 1
                elapsed = time.monotonic() - started
                timeout = self.broker.cache_wait_timeout
                if timeout and elapsed >= timeout:
                    raise RuntimeError('Best-weight candidate lease wait timed out: ' + exc.path) from exc
                self.status('waiting_best_weight', active_cid=best['candidate_id'], waiting_for=exc.path)
                atomic_json(self.run_dir / 'checkpoint_wait.json', {'state': 'waiting', 'candidate_id': best['candidate_id'], 'lock': exc.path, 'polls': waits, 'seconds': elapsed})
                self.broker.stop.wait(min(0.2, max(0.0, timeout - elapsed)) if timeout else 0.2)
        raise StopRequested()

    def committed(self, fe, best):
        if self.args.mode == 'mstar':
            self.ensure_checkpoint(best)
        self._candidate_state = None
        self.broker.advance(self.args.algorithm, self.run, fe)
        self.status('running')
