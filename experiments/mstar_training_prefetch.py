from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))

class TrainingBatches:

    def __init__(self, loader, device, fast=True):
        self.loader = loader
        self.device = device
        self.fast = fast
        self.thread = None

    def __enter__(self):
        self.iterator = iter(self.loader)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        import torch
        x, y = next(self.iterator)
        x = (x + torch.randn_like(x) * 0.01).clamp(-2.1179, 2.64)
        return (x.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last), y.to(self.device, non_blocking=True))

    def __exit__(self, *args):
        self.iterator = None

def release_candidate_memory(device, destination, cid):
    if device.type != 'cuda':
        return
    import gc, torch
    from mstar_common import atomic_json
    gc.collect()
    torch.cuda.synchronize(device)
    before = {'allocated_mib': torch.cuda.memory_allocated(device) / 2 ** 20, 'reserved_mib': torch.cuda.memory_reserved(device) / 2 ** 20}
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    atomic_json(destination, {'candidate_id': cid, 'before_cleanup': before, 'after_cleanup': {'allocated_mib': torch.cuda.memory_allocated(device) / 2 ** 20, 'reserved_mib': torch.cuda.memory_reserved(device) / 2 ** 20}, 'peak_allocated_mib': peak, 'device_free_mib': free / 2 ** 20, 'device_total_mib': total / 2 ** 20, 'scope': 'PyTorch allocated/reserved; device_free includes other processes'})
