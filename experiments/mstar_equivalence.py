from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import importlib.util, gc
from pathlib import Path
from types import SimpleNamespace
from mstar_common import atomic_json, digest, read_json, runtime_hashes, seed_for, decode_candidate
from mstar_migration_v1 import reference_hashes
BASE = Path(__file__).parent

def load_reference():
    reference_hashes()
    spec = importlib.util.spec_from_file_location('frozen_v1_cnn_core', BASE / 'mstar_v1_cnn_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def certify(args, hardware, samples, ui):
    import torch, mstar_cnn_core as core
    from mstar_tensor_cache import open_cache
    from src.HGA.mstar.checkpoints import state_digest
    identity = {'dataset_id': args.dataset_id, 'hardware': hardware, 'environment': args.environment, 'runtime_hashes': runtime_hashes(), 'reference_hashes': reference_hashes(), 'probe_cid': 5696, 'epochs': 1}
    destination = Path(args.performance_cache) / ('equivalence_' + digest(identity) + '.json')
    if destination.exists():
        obj = read_json(destination)
        clean = dict(obj)
        checksum = clean.pop('checksum', None)
        if checksum != digest(clean) or obj['identity'] != identity or obj['passed'] is not True:
            raise ValueError('Equivalence certificate checksum/identity differs')
        atomic_json(Path(args.output) / 'equivalence.json', obj)
        return obj
    old = load_reference()
    device = torch.device('cuda:0')
    torch.use_deterministic_algorithms(True)
    core.configure_performance(args)
    old.configure_performance(args)
    pretrained = torch.hub.load_state_dict_from_url(core.models.ResNet18_Weights.DEFAULT.url, progress=False, map_location='cpu', check_hash=True)

    def builder(mode):
        m = core.models.resnet18(weights=None)
        m.load_state_dict(pretrained)
        m.fc = core.nn.Linear(m.fc.in_features, 10)
        return m
    old.build_resnet18 = builder
    core.build_resnet18 = builder
    cfg = decode_candidate(5696)
    cfg['epochs'] = 1
    seed = seed_for('candidate', args.seed, 0, 5696)
    results = []
    for module, label in [(old, 'V1 reference'), (core, 'V3 sequential')]:
        ui.update(11, 'Verify training equivalence', above=[label + ' | one epoch | train/validation only'], force=True)
        if module is core:
            core.FAST_TRAINING = True
            core.VAL_JOBS = args.jobs
            core.GLOBAL_TENSOR_CACHE = open_cache(args.tensor_cache, args.dataset_id)
        module.GLOBAL_VAL_STORE = module.EvalTensorStore(samples['validation'], None, device, 'auto')
        result = module.train_candidate(cfg, samples['train'], samples['validation'], device, 0, seed, 'imagenet')
        results.append({'f1': result['best_val_macro_f1'], 'loss': result['best_val_loss'], 'epoch': result['best_epoch'], 'history': result['epoch_history'], 'state_sha256': state_digest(result['best_state'])})
        del result
        module.GLOBAL_VAL_STORE = None
        gc.collect()
        torch.cuda.empty_cache()
    core.GLOBAL_TENSOR_CACHE = None
    if results[0] != results[1]:
        atomic_json(Path(args.output) / 'equivalence_failure.json', {'identity': identity, 'reference': results[0], 'optimized': results[1]})
        raise ValueError('V1/V3 training equivalence differs; no V1 import or search was performed')
    obj = {'identity': identity, 'passed': True, 'result': results[0], 'scope': 'Real ResNet18/real train+validation, one candidate/one epoch. Not a full-budget or speed guarantee.'}
    obj['checksum'] = digest(obj)
    atomic_json(destination, obj)
    atomic_json(Path(args.output) / 'equivalence.json', obj)
    return obj
