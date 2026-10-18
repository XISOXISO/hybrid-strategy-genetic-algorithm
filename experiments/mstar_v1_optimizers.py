from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
from mstar_v1_de_optimizers import SuccessHistoryDE

def run_optimizer(name, journal, seed, budget, population=10):
    if name not in ('JSO', 'LSHADE'):
        raise ValueError('Invalid optimizer name')
    engine = SuccessHistoryDE(name, journal, seed, budget)
    engine.run()
    return engine
