from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import shutil
import sys
import time

class Dashboard:

    def __init__(self, stream=None, refresh=1.0):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.refresh = refresh
        self.last = 0.0
        self.lines = 0
        self.started = time.monotonic()
        self.percent = 0.0

    def update(self, percent, stage, above=(), below=(), force=False):
        now = time.monotonic()
        self.percent = max(self.percent, min(100.0, float(percent)))
        if not force and now - self.last < self.refresh:
            return
        self.last = now
        width = max(50, min(110, shutil.get_terminal_size((100, 24)).columns - 2))
        bar_width = max(12, width - 27)
        full = int(bar_width * self.percent / 100)
        elapsed = int(now - self.started)
        elapsed_text = f'{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}'
        bar = f"[{('#' * full).ljust(bar_width, '-')}] {self.percent:6.2f}% {elapsed_text}"
        top = ['ATRNet-STAR | SOC10 N4800 | formal v5 B500 + final test', f'Stage: {stage}']
        top += list(above)[:2]
        top += [''] * (4 - len(top))
        bottom = list(below)
        row_count = max(9, self.lines, 5 + len(bottom))
        bottom += [''] * (row_count - 5 - len(bottom))
        rows = [line[:width] for line in top + [bar] + bottom]
        if self.tty:
            if self.lines:
                self.stream.write(f'\x1b[{self.lines}A')
            for line in rows:
                self.stream.write('\r\x1b[2K' + line + '\n')
            self.lines = len(rows)
        else:
            info = next((x for x in below if x), '')
            self.stream.write('\r' + bar + ' | ' + stage + ' | ' + info + ' ' * 8)
        self.stream.flush()

    def close(self):
        if not self.tty:
            self.stream.write('\n')
            self.stream.flush()
