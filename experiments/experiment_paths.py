from pathlib import Path
import sys


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
for path in (REPOSITORY_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def algorithm_source_paths(task):
    result = {}
    for family in ('HGA', 'baselines'):
        directory = REPOSITORY_ROOT / 'src' / family / task
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing algorithm directory: {directory}')
        for path in sorted(directory.glob('*.py')):
            result[path.relative_to(REPOSITORY_ROOT).as_posix()] = path
    return result
