from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, os, time, random, math, hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from atrnet_star_atr_image import read_sample_uint8
NUM_CLASSES = 10
NUMERICAL_FAILURE_F1 = 0.0
NUMERICAL_FAILURE_ACCURACY = 0.0
NUMERICAL_FAILURE_LOSS = 10.0
PERF = {}
GLOBAL_VAL_STORE = None
GLOBAL_RAM_IMAGE_CACHE = None
IMAGE_CACHE_PATH_MEMO = {}

class _NoGovernor:

    def gpu_begin(self, device):
        return None

    def gpu_end(self, token, device):
        pass
RESOURCE_GOVERNOR = _NoGovernor()

def set_seed(seed: int) -> None:
    seed = int(seed) % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def finite_float(value: Any, fallback: float) -> float:
    try:
        x = float(value)
    except Exception:
        return float(fallback)
    return x if math.isfinite(x) else float(fallback)

def finite_f1(value: Any) -> float:
    return min(1.0, max(0.0, finite_float(value, NUMERICAL_FAILURE_F1)))

def finite_loss(value: Any) -> float:
    x = finite_float(value, NUMERICAL_FAILURE_LOSS)
    return max(0.0, x)

def configure_performance(args: argparse.Namespace) -> None:
    mode = str(args.performance_mode)
    if mode not in {'safe', 'compat', 'fast'}:
        raise ValueError('--performance-mode must be safe/compat/fast')
    PERF['channels_last'] = bool(mode != 'compat')
    PERF['gpu_metrics'] = bool(mode != 'compat')
    PERF['gpu_noise'] = bool(mode != 'compat')
    PERF['prefetch_factor'] = max(1, int(args.prefetch_factor))
    PERF['val_cache_mode'] = str(args.val_cache)
    PERF['val_gpu_fraction'] = float(args.val_gpu_fraction)
    PERF['ram_image_cache_mode'] = str(args.ram_image_cache)
    PERF['clear_cuda_cache_every'] = max(0, int(args.clear_cuda_cache_every))
    if torch.cuda.is_available():
        if mode == 'fast':
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

def _maybe_channels_last(x: torch.Tensor) -> torch.Tensor:
    if bool(PERF.get('channels_last', False)) and x.ndim == 4 and x.is_cuda:
        return x.contiguous(memory_format=torch.channels_last)
    return x

def _model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if device.type == 'cuda' and bool(PERF.get('channels_last', False)):
        model = model.to(memory_format=torch.channels_last)
    return model

def _confusion_update(conf: torch.Tensor, y: torch.Tensor, pred: torch.Tensor) -> None:
    idx = y.to(torch.int64) * NUM_CLASSES + pred.to(torch.int64)
    conf.add_(torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES))

def _metrics_from_confusion(conf: torch.Tensor) -> Dict[str, Any]:
    c = conf.to(torch.float64)
    tp = torch.diag(c)
    support = c.sum(dim=1)
    pred_count = c.sum(dim=0)
    fp = pred_count - tp
    fn = support - tp
    denom = 2.0 * tp + fp + fn
    f1 = torch.where(denom > 0, 2.0 * tp / denom, torch.zeros_like(denom))
    present = support > 0
    macro_present = f1[present].mean() if bool(present.any()) else torch.tensor(0.0, dtype=torch.float64, device=c.device)
    macro_all = f1.mean()
    total = support.sum()
    accuracy = tp.sum() / total.clamp_min(1.0)
    weighted = (f1 * support).sum() / total.clamp_min(1.0)
    return {'accuracy': float(accuracy.item()), 'macro_f1': float(macro_present.item()), 'macro_f1_present': float(macro_present.item()), 'macro_f1_all_train_classes': float(macro_all.item()), 'weighted_f1': float(weighted.item()), 'present_labels': torch.nonzero(present, as_tuple=False).reshape(-1).detach().cpu().tolist()}

def image_cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|atr-released-uint8-v1'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def memoized_cache_path(path: Path, cache_dir: Optional[Path]) -> Optional[Path]:
    if cache_dir is None:
        return None
    key = (str(cache_dir), str(path))
    cached_text = IMAGE_CACHE_PATH_MEMO.get(key)
    if cached_text is None and key not in IMAGE_CACHE_PATH_MEMO:
        cached_text = str(cache_dir / f'{image_cache_key(path)}.npy')
        IMAGE_CACHE_PATH_MEMO[key] = cached_text
    return Path(cached_text) if cached_text is not None else None

def load_cached_uint8(path: Path, cached: Optional[Path]) -> np.ndarray:
    key = str(path)
    global GLOBAL_RAM_IMAGE_CACHE
    if GLOBAL_RAM_IMAGE_CACHE is not None and key in GLOBAL_RAM_IMAGE_CACHE:
        return GLOBAL_RAM_IMAGE_CACHE[key]
    if cached is not None and cached.is_file():
        return np.load(cached, allow_pickle=False)
    array = read_sample_uint8(path)
    if cached is not None and (not cached.exists()):
        tmp = cached.with_name(cached.name + f'.{os.getpid()}.tmp')
        try:
            with tmp.open('wb') as h:
                np.save(h, array, allow_pickle=False)
            os.replace(tmp, cached)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
    return array

def preload_raw_image_cache(samples: Sequence[Tuple[str, int]], cache_dir: Optional[Path]) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    unique_paths = list(dict.fromkeys((str(x[0]) for x in samples)))
    print(f'Preloading {len(unique_paths)} decoded uint8 images into CPU RAM ...')
    started = time.perf_counter()
    total_bytes = 0
    for i, path_text in enumerate(unique_paths, 1):
        path = Path(path_text)
        cached = memoized_cache_path(path, cache_dir)
        arr = load_cached_uint8(path, cached)
        cache[path_text] = arr
        total_bytes += int(arr.nbytes)
        if i % 1000 == 0:
            print(f'  RAM image cache: {i}/{len(unique_paths)}')
    print(f'RAM image cache ready: {total_bytes / 1024 ** 2:.1f} MiB in {time.perf_counter() - started:.1f}s')
    return cache

class MSTARImageDataset(Dataset):

    def __init__(self, samples: Sequence[Tuple[str, int]], transform, cache_dir: Optional[Path]):
        self.samples = list(samples)
        self.transform = transform
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_paths: List[Optional[str]] = []
        for path_text, _ in self.samples:
            path = Path(path_text)
            cp = memoized_cache_path(path, cache_dir)
            self.cached_paths.append(str(cp) if cp is not None else None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path_text, label = self.samples[index]
        path = Path(path_text)
        cp = self.cached_paths[index]
        cached = Path(cp) if cp is not None else None
        array = load_cached_uint8(path, cached)
        image = Image.fromarray(array, mode='L').convert('RGB')
        return (self.transform(image), int(label))

class EvalTensorStore:

    def __init__(self, samples: Sequence[Tuple[str, int]], cache_dir: Optional[Path], device: torch.device, mode: str):
        _, eval_tf, _ = build_transforms(1.0)
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        print(f'Preprocessing validation set once ({len(samples)} samples) ...')
        started = time.perf_counter()
        for i, (path_text, label) in enumerate(samples, 1):
            path = Path(path_text)
            cached = memoized_cache_path(path, cache_dir)
            arr = load_cached_uint8(path, cached)
            img = Image.fromarray(arr, mode='L').convert('RGB')
            xs.append(eval_tf(img))
            ys.append(int(label))
            if i % 1000 == 0:
                print(f'  validation preprocess: {i}/{len(samples)}')
        self.x = torch.stack(xs, dim=0).contiguous()
        self.y = torch.tensor(ys, dtype=torch.long)
        self.location = 'cpu'
        self.pinned = False
        mb = (self.x.numel() * self.x.element_size() + self.y.numel() * self.y.element_size()) / 1024 ** 2
        target = mode
        if mode == 'auto':
            target = 'cpu'
            if device.type == 'cuda':
                try:
                    free_b, _ = torch.cuda.mem_get_info(device)
                    if mb * 1024 ** 2 <= float(PERF.get('val_gpu_fraction', 0.18)) * free_b:
                        target = 'gpu'
                except Exception:
                    target = 'cpu'
        if target == 'gpu' and device.type == 'cuda':
            self.x = self.x.to(device, non_blocking=False)
            self.y = self.y.to(device, non_blocking=False)
            if bool(PERF.get('channels_last', False)):
                self.x = self.x.contiguous(memory_format=torch.channels_last)
            self.location = 'gpu'
        elif device.type == 'cuda':
            try:
                self.x = self.x.pin_memory()
                self.y = self.y.pin_memory()
                self.pinned = True
            except Exception:
                self.pinned = False
        print(f'Validation tensor cache: {mb:.1f} MiB, location={self.location}, built in {time.perf_counter() - started:.1f}s')

    def evaluate(self, model: nn.Module, criterion, device: torch.device, batch_size: int) -> Dict[str, Any]:
        model.eval()
        conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        n = int(self.y.numel())
        with torch.inference_mode():
            for start in range(0, n, int(batch_size)):
                _gov_token = RESOURCE_GOVERNOR.gpu_begin(device)
                end = min(n, start + int(batch_size))
                xb = self.x[start:end]
                yb = self.y[start:end]
                if self.location != 'gpu':
                    xb = xb.to(device, non_blocking=self.pinned)
                    yb = yb.to(device, non_blocking=self.pinned)
                    xb = _maybe_channels_last(xb)
                if device.type == 'cuda':
                    logits, loss = _bf16_forward_fp32_loss(model, xb, yb, criterion)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                total_loss += loss.detach().to(torch.float64) * xb.size(0)
                _confusion_update(conf, yb, logits.argmax(1))
                RESOURCE_GOVERNOR.gpu_end(_gov_token, device)
        mean_loss = float((total_loss / max(1, n)).item())
        if not math.isfinite(mean_loss):
            return {'accuracy': NUMERICAL_FAILURE_ACCURACY, 'macro_f1': NUMERICAL_FAILURE_F1, 'weighted_f1': NUMERICAL_FAILURE_F1, 'loss': NUMERICAL_FAILURE_LOSS, 'nonfinite': 1}
        metrics = _metrics_from_confusion(conf)
        metrics['loss'] = finite_loss(mean_loss)
        metrics['nonfinite'] = 0
        return metrics

def build_transforms(augmentation_strength: float):
    strength = float(augmentation_strength)
    if abs(strength - 0.5) < 1e-12:
        min_scale = 0.95
    elif abs(strength - 1.0) < 1e-12:
        min_scale = 0.9
    elif abs(strength - 1.5) < 1e-12:
        min_scale = 0.85
    else:
        raise ValueError(f'Unsupported augmentation strength: {strength}')
    rotation = 5.0 * strength
    noise_std = 0.01 * strength
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomResizedCrop(224, scale=(min_scale, 1.0), ratio=(0.95, 1.05)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(rotation), transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    return (train_tf, eval_tf, float(noise_std))

def build_resnet18(weights_mode: str) -> nn.Module:
    if weights_mode == 'imagenet':
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=False)
        except Exception as exc:
            raise RuntimeError('Unable to load ImageNet-pretrained ResNet18. Pre-download the weights; do not change weights mode for a resumed experiment.') from exc
    elif weights_mode == 'none':
        model = models.resnet18(weights=None)
    else:
        raise ValueError('--weights must be imagenet or none')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def make_optimizer(model: nn.Module, config: Mapping[str, Any]):
    name, lr, wd = (str(config['optimizer']), float(config['learning_rate']), float(config['weight_decay']))
    if name == 'SGD':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if name == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'RMSprop':
        return RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise KeyError(name)

def make_scheduler(optimizer, config: Mapping[str, Any]):
    name, p = (str(config['scheduler']), dict(config['scheduler_params']))
    if name == 'StepLR':
        return StepLR(optimizer, step_size=int(p['step_size']), gamma=float(p['gamma']))
    if name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=int(p['T_max']), eta_min=float(p['eta_min']))
    if name == 'ExponentialLR':
        return ExponentialLR(optimizer, gamma=float(p['gamma']))
    raise KeyError(name)

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device, seed: int):
    kwargs = dict(batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers), pin_memory=device.type == 'cuda', drop_last=False)
    if int(num_workers) > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = int(PERF.get('prefetch_factor', 2))
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs['generator'] = generator
    return DataLoader(dataset, **kwargs)

def _bf16_forward_fp32_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(x)
    loss = criterion(logits.float(), y)
    return (logits, loss)

def evaluate_loader(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Dict[str, Any]:
    global GLOBAL_VAL_STORE
    if GLOBAL_VAL_STORE is not None:
        return GLOBAL_VAL_STORE.evaluate(model, criterion, device, batch_size=getattr(loader, 'batch_size', 64) or 64)
    model.eval()
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            _gov_token = RESOURCE_GOVERNOR.gpu_begin(device)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = _maybe_channels_last(x)
            if device.type == 'cuda':
                logits, loss = _bf16_forward_fp32_loss(model, x, y, criterion)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.detach().to(torch.float64) * x.size(0)
            n += int(x.size(0))
            _confusion_update(conf, y, logits.argmax(1))
            RESOURCE_GOVERNOR.gpu_end(_gov_token, device)
    mean_loss = float((total_loss / max(1, n)).item())
    if not math.isfinite(mean_loss):
        return {'accuracy': NUMERICAL_FAILURE_ACCURACY, 'macro_f1': NUMERICAL_FAILURE_F1, 'weighted_f1': NUMERICAL_FAILURE_F1, 'loss': NUMERICAL_FAILURE_LOSS, 'nonfinite': 1}
    metrics = _metrics_from_confusion(conf)
    metrics['loss'] = finite_loss(mean_loss)
    metrics['nonfinite'] = 0
    return metrics

def train_candidate(*, config: Mapping[str, Any], train_samples: Sequence[Tuple[str, int]], val_samples: Sequence[Tuple[str, int]], device: torch.device, num_workers: int, seed: int, weights_mode: str, cache_dir: Optional[Path], return_best_state: bool=False, collect_train_metrics: bool=True, progress_callback=None, epoch_prune_callback=None) -> Dict[str, Any]:
    set_seed(seed)
    train_tf, eval_tf, noise_std = build_transforms(float(config['augmentation_strength']))
    train_ds = MSTARImageDataset(train_samples, train_tf, cache_dir)
    val_ds = MSTARImageDataset(val_samples, eval_tf, cache_dir) if GLOBAL_VAL_STORE is None else None
    train_loader = make_loader(train_ds, int(config['batch_size']), True, num_workers, device, seed)
    val_loader = make_loader(val_ds, max(32, int(config['batch_size'])), False, num_workers, device, seed) if val_ds is not None else None
    model = _model_to_device(build_resnet18(weights_mode), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    best_f1, best_acc, best_loss, best_epoch, best_state = (-1.0, -1.0, float('inf'), -1, None)
    epoch_history: List[Dict[str, Any]] = []
    numerical_failure = False
    pruned = False
    epochs_completed = 0
    numerical_failure_reason = ''
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_progress = 0.0
    for epoch in range(1, int(config['epochs']) + 1):
        if progress_callback:
            progress_callback('train', epoch, int(config['epochs']), 0, len(train_loader), None)
        model.train()
        total = torch.zeros((), dtype=torch.float64, device=device)
        train_conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device) if collect_train_metrics else None
        train_n = 0
        for batch_index, (x, y) in enumerate(train_loader, 1):
            _gov_token = RESOURCE_GOVERNOR.gpu_begin(device)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = _maybe_channels_last(x)
            if noise_std > 0:
                if device.type == 'cuda' and bool(PERF.get('gpu_noise', True)):
                    x.add_(torch.randn_like(x) * noise_std).clamp_(-2.1179, 2.64)
                else:
                    x = torch.clamp(x + torch.randn_like(x) * noise_std, -2.1179, 2.64)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                logits, loss = _bf16_forward_fp32_loss(model, x, y, criterion)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.detach().to(torch.float64) * x.size(0)
            train_n += int(x.size(0))
            if collect_train_metrics:
                _confusion_update(train_conf, y, logits.detach().argmax(1))
            now = time.monotonic()
            if progress_callback and (now - last_progress >= 2 or batch_index == len(train_loader)):
                progress_callback('train', epoch, int(config['epochs']), batch_index, len(train_loader), None)
                last_progress = now
            RESOURCE_GOVERNOR.gpu_end(_gov_token, device)
        if numerical_failure:
            break
        scheduler.step()
        train_loss_value = float((total / max(1, train_n)).item())
        if not math.isfinite(train_loss_value):
            numerical_failure = True
            numerical_failure_reason = f'non-finite train loss at epoch {epoch}'
            print(f"      WARNING: {numerical_failure_reason}; candidate={config.get('candidate_id')} seed={seed}")
            break
        if progress_callback:
            progress_callback('validation', epoch, int(config['epochs']), 0, 0, None)
        if GLOBAL_VAL_STORE is not None:
            val = GLOBAL_VAL_STORE.evaluate(model, criterion, device, batch_size=max(32, int(config['batch_size'])))
        else:
            val = evaluate_loader(model, val_loader, criterion, device)
        train_metrics = _metrics_from_confusion(train_conf) if collect_train_metrics else None
        current_f1 = finite_f1(val.get('macro_f1'))
        current_acc = finite_f1(val.get('accuracy'))
        current_loss = finite_loss(val.get('loss'))
        val_nonfinite = bool(int(val.get('nonfinite', 0)))
        epoch_history.append({'epoch': epoch, 'train_loss': train_loss_value, 'train_accuracy': float(train_metrics['accuracy']) if train_metrics else None, 'train_macro_f1': float(train_metrics['macro_f1']) if train_metrics else None, 'val_loss': current_loss, 'val_accuracy': current_acc, 'val_macro_f1': current_f1, 'val_nonfinite': int(val_nonfinite)})
        epochs_completed = epoch
        if progress_callback:
            progress_callback('epoch_done', epoch, int(config['epochs']), 0, 0, current_f1)
        if val_nonfinite:
            numerical_failure = True
            numerical_failure_reason = f'non-finite validation loss at epoch {epoch}'
            print(f"      WARNING: {numerical_failure_reason}; candidate={config.get('candidate_id')} seed={seed}")
            break
        if current_f1 > best_f1 + 1e-12 or (abs(current_f1 - best_f1) <= 1e-12 and current_loss < best_loss):
            best_f1, best_acc, best_loss, best_epoch = (current_f1, current_acc, current_loss, epoch)
            if return_best_state:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch_prune_callback and epoch_prune_callback(epoch, current_f1, current_loss):
            pruned = True
            break
    seconds = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == 'cuda' else 0.0
    if best_epoch < 0 or not (math.isfinite(best_f1) and math.isfinite(best_loss)):
        best_f1 = NUMERICAL_FAILURE_F1
        best_acc = NUMERICAL_FAILURE_ACCURACY
        best_loss = NUMERICAL_FAILURE_LOSS
        best_epoch = 0
        numerical_failure = True
        if not numerical_failure_reason:
            numerical_failure_reason = 'no finite validation epoch'
    result = {'best_val_macro_f1': finite_f1(best_f1), 'best_val_accuracy': finite_f1(best_acc), 'best_val_loss': finite_loss(best_loss), 'best_epoch': int(best_epoch), 'training_seconds': finite_float(seconds, 0.0), 'peak_vram_mb': finite_float(peak, 0.0), 'numerical_failure': int(numerical_failure), 'numerical_failure_reason': str(numerical_failure_reason), 'precision_mode': 'bf16' if device.type == 'cuda' else 'fp32_cpu', 'epoch_history': epoch_history, 'pruned': pruned, 'epochs_completed': epochs_completed}
    if return_best_state:
        result['model'] = model
        result['best_state'] = best_state
    else:
        del model
    del criterion, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    return result
