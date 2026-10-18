from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import csv, json, os, subprocess, threading, time, uuid
from contextlib import contextmanager
from pathlib import Path
from mstar_common import atomic_json, digest, read_json

def audit_contract(interval=2.0, measurement_mode='shared'):
    return {'version': 1, 'resource_interval_seconds': interval, 'measurement_mode': measurement_mode, 'checkpoint_budgets': list(range(50, 501, 50)), 'timing': 'sum worker attempts, includes replay/loading/checkpoint IO; excludes queue, preflight and offline pauses', 'GPU_utilization': 'whole visible device; concurrent tracks cannot be attributed individually', 'historical_missing': 'not reconstructed', 'test_curve': 'offline only after selections frozen; no search feedback'}

def read_audit(path):
    path = Path(path)
    if not path.exists():
        return ([], False)
    data = path.read_bytes()
    lines = data.splitlines()
    torn = bool(data and (not data.endswith(b'\n')))
    if torn:
        lines = lines[:-1]
    records = []
    for line in lines:
        r = json.loads(line)
        c = dict(r)
        checksum = c.pop('checksum', None)
        if checksum != digest(c):
            raise ValueError('Audit event checksum mismatch')
        records.append(r)
    return (records, torn)

class AttemptAudit:

    def __init__(self, directory, args, committed_fe, stage='search'):
        self.rd = Path(directory)
        self.args = args
        self.stage = stage
        self.path = self.rd / 'audit_events.jsonl'
        self.start = time.monotonic()
        self.attempt_id = uuid.uuid4().hex
        self.launch_id = getattr(args, 'launch_id', 'standalone')
        self.active = {'attempt_id': self.attempt_id, 'launch_id': self.launch_id, 'stage': stage, 'pid': os.getpid(), 'algorithm': args.algorithm, 'run': args.run, 'phase': 'replay'}
        origin = self.rd / 'audit_origin.json'
        if not origin.exists():
            atomic_json(origin, {'protocol_id': args.protocol_id, 'observed_from_fe': committed_fe, 'historical_timing_complete': committed_fe == 0, 'historical_resources_complete': committed_fe == 0})
        elif read_json(origin)['protocol_id'] != args.protocol_id:
            raise ValueError('Audit origin protocol mismatch')
        _, torn = read_audit(self.path)
        if torn:
            data = self.path.read_bytes()
            self.path.write_bytes(data[:data.rfind(b'\n') + 1])
        self.emit('begin', committed_fe=committed_fe, previous_torn_event=torn)
        atomic_json(self.rd / 'audit_active.json', self.active)

    def emit(self, event, **extra):
        record = {**self.active, 'event': event, 'utc': time.time(), **extra}
        record['checksum'] = digest(record)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    @contextmanager
    def phase(self, name, **fields):
        before = self.active['phase']
        self.active['phase'] = name
        atomic_json(self.rd / 'audit_active.json', self.active)
        start = time.monotonic()
        result = 'completed'
        try:
            yield
        except BaseException:
            result = 'interrupted_or_failed'
            raise
        finally:
            self.emit('phase', phase=name, seconds=time.monotonic() - start, status=result, **fields)
            self.active['phase'] = before
            atomic_json(self.rd / 'audit_active.json', self.active)

    def end(self, status, committed_fe):
        self.emit('end', seconds=time.monotonic() - self.start, status=status, committed_fe=committed_fe)
        self.active['phase'] = 'closed'
        atomic_json(self.rd / 'audit_active.json', self.active)

def cpu_capacity():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            count = min(count, int(quota) / int(period))
    except (OSError, ValueError):
        for folder in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
            try:
                quota = int((Path(folder) / 'cpu.cfs_quota_us').read_text())
                period = int((Path(folder) / 'cpu.cfs_period_us').read_text())
                if quota > 0:
                    count = min(count, quota / period)
                break
            except (OSError, ValueError, ZeroDivisionError):
                continue
    return max(0.01, float(count))

def process_usage(pid):
    own = Path('/proc/self/stat').read_text()
    if pid == os.getpid():
        text = own
    else:
        if int(own.split(' ', 1)[0]) != os.getpid():
            raise OSError('PID namespace differs from /proc; process resources unavailable')
        text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return ((float(fields[11]) + float(fields[12])) / os.sysconf('SC_CLK_TCK'), int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 1024 ** 2, fields[19])

def gpu_usage():

    def query(kind, fields):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}=' + fields, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    rows = query('gpu', 'uuid,utilization.gpu,memory.used,memory.total')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
    selected = next((r for r in rows if r[0] == visible), None)
    if selected is None and visible.isdigit() and (int(visible) < len(rows)):
        selected = rows[int(visible)]
    if selected is None and len(rows) == 1:
        selected = rows[0]
    if selected is None:
        raise ValueError('Cannot identify the visible GPU for sampling')
    gpu = {'gpu_uuid': selected[0], 'device_gpu_percent': float(selected[1]), 'device_vram_mib': float(selected[2]), 'device_vram_total_mib': float(selected[3])}
    gpu['device_vram_percent'] = 100 * gpu['device_vram_mib'] / gpu['device_vram_total_mib']
    processes = {}
    process_error = ''
    try:
        for pid, memory, gid in query('compute-apps', 'pid,used_memory,gpu_uuid'):
            if gid == selected[0]:
                processes[int(pid)] = float(memory)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        process_error = str(exc)
    return (gpu, processes, process_error)
FIELDS = ['utc', 'monotonic', 'launch_id', 'stage', 'algorithm', 'run', 'pid', 'attempt_id', 'phase', 'sample_interval_seconds', 'cpu_cores_percent', 'cpu_allocation_percent', 'cpu_capacity', 'rss_mib', 'process_vram_mib', 'gpu_uuid', 'device_gpu_percent', 'device_vram_mib', 'device_vram_total_mib', 'device_vram_percent', 'active_tracks', 'gpu_scope', 'sampling_error']

class ResourceSampler:

    def __init__(self, output, launch_id, interval=2.0, gpu=True):
        self.output = Path(output)
        self.launch_id = launch_id
        self.interval = interval
        self.gpu = gpu
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.tracks = []
        self.previous = {}
        self.capacity = cpu_capacity()
        self.thread = threading.Thread(target=self.loop, name='resource-sampler', daemon=True)
        self.thread.start()

    def update(self, scheduler):
        tracks = [(key, state['process'].pid, scheduler.stage) for key, state in scheduler.active.items()]
        with self.lock:
            self.tracks = tracks

    @staticmethod
    def append(path, row):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or not path.stat().st_size
        with path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k, '') for k in FIELDS})
            f.flush()

    def sample(self):
        with self.lock:
            tracks = list(self.tracks)
        if not tracks:
            return
        gpu = {}
        memory = {}
        error = ''
        if self.gpu:
            try:
                gpu, memory, error = gpu_usage()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                error = str(exc)
        now = time.monotonic()
        utc = time.time()
        global_row = {'utc': utc, 'monotonic': now, 'launch_id': self.launch_id, 'stage': tracks[0][2], **gpu, 'active_tracks': len(tracks), 'gpu_scope': 'whole_device', 'sampling_error': error}
        self.append(self.output / 'device_resources.csv', global_row)
        for (name, run), pid, stage in tracks:
            rd = self.output / 'runs' / name / f'run_{run:02d}'
            try:
                active = read_json(rd / 'audit_active.json') if (rd / 'audit_active.json').exists() else {}
                if active.get('pid') != pid or active.get('launch_id') != self.launch_id:
                    active = {}
                used, rss, birth = process_usage(pid)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.append(rd / 'resource_samples.csv', {**global_row, 'stage': stage, 'algorithm': name, 'run': run, 'pid': pid, 'cpu_capacity': self.capacity, 'sampling_error': ' | '.join((s for s in [error, str(exc)] if s)), 'gpu_scope': 'shared_device' if len(tracks) > 1 else 'device_during_single_track'})
                continue
            previous = self.previous.get((pid, birth))
            dt = now - previous[0] if previous else 0.0
            cpu = 100 * max(0.0, used - previous[1]) / dt if dt > 0 else None
            self.previous[pid, birth] = (now, used)
            row = {**global_row, 'stage': stage, 'algorithm': name, 'run': run, 'pid': pid, 'attempt_id': active.get('attempt_id', ''), 'phase': active.get('phase', 'starting'), 'sample_interval_seconds': dt if dt else '', 'cpu_cores_percent': cpu if cpu is not None else '', 'cpu_allocation_percent': cpu / self.capacity if cpu is not None else '', 'cpu_capacity': self.capacity, 'rss_mib': rss, 'process_vram_mib': memory.get(pid, ''), 'gpu_scope': 'shared_device' if len(tracks) > 1 else 'device_during_single_track'}
            self.append(rd / 'resource_samples.csv', row)

    def loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                atomic_json(self.output / 'resource_sampling_error.json', {'utc': time.time(), 'error': str(exc)})

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError('Resource sampler did not stop')

def timing_summary(directory):
    rd = Path(directory)
    events, torn = read_audit(rd / 'audit_events.jsonl')
    origin = read_json(rd / 'audit_origin.json') if (rd / 'audit_origin.json').exists() else {'observed_from_fe': None, 'historical_timing_complete': False}
    begins = {e['attempt_id']: e for e in events if e['event'] == 'begin' and e['stage'] == 'search'}
    ends = {e['attempt_id']: e for e in events if e['event'] == 'end' and e['stage'] == 'search'}
    unmatched = set(begins) - set(ends)
    phases = [e for e in events if e['event'] == 'phase' and e['stage'] == 'search']
    complete = origin['historical_timing_complete'] and (not unmatched) and (not torn) and (not any((e.get('previous_torn_event') for e in begins.values())))
    return {'observed_from_fe': origin['observed_from_fe'], 'timing_complete_from_fe0': complete, 'closed_attempts': len(ends), 'unclosed_attempts': len(unmatched), 'observed_active_wall_seconds': sum((e['seconds'] for e in ends.values())), 'observed_training_phase_seconds': sum((e['seconds'] for e in phases if e['phase'] == 'train')), 'observed_checkpoint_phase_seconds': sum((e['seconds'] for e in phases if e['phase'] == 'checkpoint')), 'observed_archive_phase_seconds': sum((e['seconds'] for e in phases if e['phase'] == 'archive')), 'historical_timing_complete': origin['historical_timing_complete'], 'note': 'Partial/lower-bound if historical FE or killed attempts are missing; excludes queue/preflight/offline pauses'}
