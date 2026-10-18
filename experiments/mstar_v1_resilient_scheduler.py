from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import errno, json, multiprocessing, os, time, traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from mstar_v1_common import atomic_json
from mstar_v1_shared_resources import available
ORDER = ('JSO', 'LSHADE', 'HGA_RM', 'HGA_CM', 'HGA_SW60', 'HGA_SW50')
POLICY = 'JSO/LSHADE -> RM -> CM -> SW60 -> SW50; nonpreemptive, no FE barrier'

def priority(key):
    name, run = key
    if name not in ORDER:
        raise ValueError('Unknown scheduled algorithm: ' + name)
    if name in ORDER[:2]:
        return (0, run, ORDER.index(name))
    return (ORDER.index(name) - 1, run, 0)

def describe_error(exc):
    chain = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append({'type': type(exc).__name__, 'message': str(exc), 'errno': getattr(exc, 'errno', None)})
        exc = exc.__cause__ or exc.__context__
    return {'chain': chain, 'traceback': traceback.format_exc()}

def recoverable(error):
    chain = error.get('chain', [])
    message = ' '.join((e['message'] for e in chain)).lower()
    forbidden = ('mismatch', 'checksum', 'tamper', 'corrupt', 'malformed', 'integrity', 'protocol', 'differs', 'changed', 'proposal guard', 'invalid optimizer', 'no space left', 'disk quota', 'permission denied', 'read-only')
    if any((s in message for s in forbidden)):
        return False
    if any((e['type'] in ('ReplayMismatch', 'ValueError', 'TypeError', 'KeyError', 'AssertionError', 'FileNotFoundError', 'PermissionError', 'ImportError', 'ModuleNotFoundError') for e in chain)):
        return False
    if any((e['type'] in ('ProcessExited', 'StalledTask', 'CacheWaitTimeout', 'MemoryError', 'OutOfMemoryError') for e in chain)):
        return True
    if any((e.get('errno') in (errno.EAGAIN, errno.ENOMEM, errno.EBUSY, errno.ETIMEDOUT, errno.ECONNRESET, errno.EINTR, errno.EIO) for e in chain)):
        return True
    return any((s in message for s in ('cuda out of memory', 'cuda error:', 'cudnn_status_alloc_failed', 'cudnn_status_execution_failed', 'cublas_status_alloc_failed', 'database is locked', 'database is busy', 'resource temporarily unavailable', 'connection timed out')))

def synthetic_error(kind, message):
    return {'chain': [{'type': kind, 'message': message, 'errno': None}], 'traceback': ''}

def child_entry(connection, function, task, initializer, initargs, logfile):
    try:
        with Path(logfile).open('a', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                if initializer:
                    initializer(*initargs)
                connection.send({'ready': True})
                result = function(task)
        connection.send({'ok': True, 'result': result})
    except BaseException as exc:
        error = describe_error(exc)
        try:
            with Path(logfile).open('a', encoding='utf-8') as f:
                f.write('\n' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n' + error['traceback'])
        except OSError:
            pass
        try:
            connection.send({'ok': False, 'error': error})
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()

def stop_process(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)

def receive_messages(state, now):
    while state['message'] is None and state['pipe'].poll():
        try:
            received = state['pipe'].recv()
        except EOFError:
            break
        if received.get('ready'):
            state['ready'] = now
            state['activity'] = now
        else:
            state['message'] = received
            state['received'] = now

class TrackScheduler:

    def __init__(self, output, jobs, base, stage='search', max_retries=3, retry_delay=10.0, stall_timeout=1800.0, cache_wait_timeout=3600.0, initializer=None, initargs=(), context='spawn'):
        self.output = Path(output)
        self.jobs = jobs
        self.base = base
        self.stage = stage
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stall_timeout = stall_timeout
        self.cache_wait_timeout = cache_wait_timeout
        self.initializer = initializer
        self.initargs = initargs
        self.context = multiprocessing.get_context(context)
        self.active = {}
        self.queue = set()
        self.delayed = {}
        self.blocked = {}
        self.attempts = {}
        self.retries = {}
        self.failures = {}
        self.succeeded = []
        self.yields = 0
        self.started = time.time()
        self.history = []
        path = self.output / 'failures.json'
        if path.exists():
            from mstar_v1_common import read_json
            self.history = read_json(path).get('history', [])

    def directory(self, key):
        return self.output / 'runs' / key[0] / f'run_{key[1]:02d}'

    def snapshot(self):
        return {'stage': self.stage, 'jobs': self.jobs, 'active_tracks': [list(k) for k in self.active], 'queued_tracks': [list(k) for k in sorted(self.queue, key=priority)], 'cache_waiters': [list(k) for k in self.blocked], 'retry_waiters': [list(k) for k in self.delayed], 'failed_tracks': [list(k) for k in self.failures], 'succeeded_tracks': [list(k) for k in self.succeeded], 'yield_events_this_launch': self.yields, 'policy': POLICY, 'max_additional_retries_per_track': self.max_retries, 'retry_events_this_launch': sum(self.retries.values())}

    def save(self):
        atomic_json(self.output / 'scheduler.json', self.snapshot())
        atomic_json(self.output / 'failures.json', {'stage': self.stage, 'launch_started_utc': self.started, 'history': self.history, 'terminal_this_stage': list(self.failures.values())})

    def fail(self, key, error):
        retry = recoverable(error) and self.retries.get(key, 0) < self.max_retries
        record = {'algorithm': key[0], 'run': key[1], 'stage': self.stage, 'utc': time.time(), 'attempt': self.attempts.get(key, 0), 'retryable': recoverable(error), 'action': 'retry_from_committed_FE' if retry else 'quarantine_this_track', 'error': error}
        if retry:
            n = self.retries.get(key, 0) + 1
            self.retries[key] = n
            delay = min(300.0, self.retry_delay * 2 ** (n - 1))
            self.delayed[key] = time.monotonic() + delay
            record['retry_delay_seconds'] = delay
        else:
            self.failures[key] = record
            atomic_json(self.directory(key) / f'{self.stage}_failure.json', record)
        self.history.append(record)
        self.save()
        with (self.output / 'recovery.log').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def launch(self, key, function):
        rd = self.directory(key)
        rd.mkdir(parents=True, exist_ok=True)
        parent, child = self.context.Pipe(duplex=False)
        task = {**self.base, 'algorithm': key[0], 'run': key[1]}
        process = self.context.Process(target=child_entry, args=(child, function, task, self.initializer, self.initargs, str(rd / f'{self.stage}_worker_errors.log')))
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        now = time.monotonic()
        self.attempts[key] = self.attempts.get(key, 0) + 1
        self.active[key] = {'process': process, 'pipe': parent, 'started': now, 'activity': now, 'stamp': None, 'message': None, 'received': None, 'ready': None}
        with (self.output / 'dispatch.log').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'algorithm': key[0], 'run': key[1], 'stage': self.stage, 'tier': priority(key)[0], 'attempt': self.attempts[key], 'active_slots': len(self.active), 'jobs': self.jobs, 'pid': process.pid}) + '\n')

    def run(self, keys, function, on_update=None, recover=None):
        self.queue = set(keys)
        try:
            while self.active or self.queue or self.delayed or self.blocked:
                now = time.monotonic()
                for key, state in list(self.active.items()):
                    process = state['process']
                    pipe = state['pipe']
                    receive_messages(state, now)
                    paths = [self.directory(key) / 'status.json', self.directory(key) / 'journal.sqlite']
                    stamp = tuple((p.stat().st_mtime_ns if p.exists() else 0 for p in paths))
                    if stamp != state['stamp']:
                        state['stamp'] = stamp
                        state['activity'] = now
                    timeout = self.stall_timeout and (now - state['activity'] > self.stall_timeout if state['ready'] is not None else now - state['started'] > max(300.0, self.stall_timeout))
                    if timeout:
                        stop_process(process)
                        state['message'] = {'ok': False, 'error': synthetic_error('StalledTask', f'No status/journal activity for {self.stall_timeout}s; process terminated')}
                    elif state['received'] and now - state['received'] > 5 and process.is_alive():
                        stop_process(process)
                    if process.is_alive():
                        continue
                    receive_messages(state, now)
                    process.join()
                    pipe.close()
                    del self.active[key]
                    message = state['message'] or {'ok': False, 'error': synthetic_error('ProcessExited', f'Worker exited with code {process.exitcode} without a result')}
                    if recover:
                        try:
                            recover(key)
                        except KeyboardInterrupt:
                            raise
                        except BaseException as exc:
                            message = {'ok': False, 'error': describe_error(exc)}
                    if not message['ok']:
                        self.fail(key, message['error'])
                        continue
                    result = message['result']
                    if isinstance(result, dict) and 'blocked' in result:
                        previous = self.blocked.get(key, {})
                        self.blocked[key] = {'path': result['blocked'], 'started': previous.get('started', now)}
                        self.yields += 1
                    else:
                        self.succeeded.append(key)
                        (self.directory(key) / f'{self.stage}_failure.json').unlink(missing_ok=True)
                for key, ready in list(self.delayed.items()):
                    if now >= ready:
                        self.queue.add(key)
                        del self.delayed[key]
                for key, wait in list(self.blocked.items()):
                    if available(wait['path']):
                        self.queue.add(key)
                        del self.blocked[key]
                    elif self.cache_wait_timeout and now - wait['started'] > self.cache_wait_timeout:
                        del self.blocked[key]
                        self.fail(key, synthetic_error('CacheWaitTimeout', f'Candidate lease unavailable for {self.cache_wait_timeout}s'))
                while self.queue and len(self.active) < self.jobs:
                    key = min(self.queue, key=priority)
                    self.queue.remove(key)
                    try:
                        self.launch(key, function)
                    except OSError as exc:
                        self.fail(key, describe_error(exc))
                self.save()
                if on_update:
                    on_update(self.snapshot())
                if self.active or self.queue or self.delayed or self.blocked:
                    time.sleep(0.25)
            return {'failed': list(self.failures.values()), 'succeeded': [list(k) for k in self.succeeded], 'retry_events': sum(self.retries.values()), 'yield_events': self.yields}
        finally:
            for state in self.active.values():
                stop_process(state['process'])
                state['pipe'].close()
            self.active.clear()
