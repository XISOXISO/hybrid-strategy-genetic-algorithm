from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import math, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
GLOBAL_RAM_IMAGE_CACHE = None
GLOBAL_VAL_STORE = None

def configure_performance(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms(augmentation_strength):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(5), normalize])
    val = transforms.Compose([transforms.Resize((224, 224)), normalize])
    return (train, val, val)

def preload_raw_image_cache(samples, cache_dir=None):
    out = {}
    for path, _ in samples:
        with Image.open(path) as im:
            out[path] = im.convert('RGB').copy()
    return out

class MSTARImageDataset(Dataset):

    def __init__(self, samples, transform, cache_dir=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        if GLOBAL_RAM_IMAGE_CACHE is not None and path in GLOBAL_RAM_IMAGE_CACHE:
            im = GLOBAL_RAM_IMAGE_CACHE[path]
        else:
            with Image.open(path) as raw:
                im = raw.convert('RGB').copy()
        x = transforms.ToTensor()(transforms.Resize((224, 224))(im))
        return (self.transform(x), label)

def make_loader(dataset, batch_size, shuffle, num_workers, device, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed))

class EvalTensorStore:

    def __init__(self, samples, cache_dir, device, mode):
        _, tf, _ = build_transforms(1.0)
        ds = MSTARImageDataset(samples, tf)
        self.x = torch.stack([ds[i][0] for i in range(len(ds))])
        self.y = torch.tensor([label for _, label in samples], dtype=torch.long)
        if device.type == 'cuda':
            self.x = self.x.pin_memory()
            self.y = self.y.pin_memory()

    def batches(self, size, device):
        for i in range(0, len(self.y), size):
            yield (self.x[i:i + size].to(device, non_blocking=True), self.y[i:i + size].to(device, non_blocking=True))

def build_resnet18(weights_mode):
    if weights_mode not in ('imagenet', 'none'):
        raise ValueError('Unknown initial weights')
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_mode == 'imagenet' else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def _model_to_device(model, device):
    return model.to(device=device, memory_format=torch.channels_last)

def _maybe_channels_last(x):
    return x.contiguous(memory_format=torch.channels_last)

def _bf16_forward_fp32_loss(model, x, y, criterion):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)
    return (logits, criterion(logits.float(), y))

def macro_f1(confusion):
    tp = np.diag(confusion).astype(float)
    den = confusion.sum(0) + confusion.sum(1)
    return float(np.divide(2 * tp, den, out=np.zeros(len(tp)), where=den != 0).mean())

def train_candidate(config, train_samples, val_samples, device, num_workers, seed, weights_mode, cache_dir=None, collect_train_metrics=False, return_best_state=True, progress_callback=None, epoch_prune_callback=None):
    set_seed(seed)
    start = time.monotonic()
    epochs = int(config['epochs'])
    model = _model_to_device(build_resnet18(weights_mode), device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    train_tf, val_tf, _ = build_transforms(config['augmentation_strength'])
    loader = make_loader(MSTARImageDataset(train_samples, train_tf), config['batch_size'], True, num_workers, device, seed)
    store = GLOBAL_VAL_STORE or EvalTensorStore(val_samples, None, device, 'cpu')
    opts = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam, 'RMSprop': torch.optim.RMSprop}
    extra = {'momentum': 0.9} if config['optimizer'] == 'SGD' else {'alpha': 0.8, 'eps': 1e-08, 'momentum': 0, 'centered': False} if config['optimizer'] == 'RMSprop' else {}
    opt = opts[config['optimizer']](model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], **extra)
    schedules = {'StepLR': torch.optim.lr_scheduler.StepLR, 'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR, 'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR}
    schedule = schedules[config['scheduler']](opt, **config['scheduler_params'])
    criterion = nn.CrossEntropyLoss()
    best = None
    best_state = None
    history = []
    last_notice = 0.0

    def notice(phase, ep, batch, batches, score=None, force=False):
        nonlocal last_notice
        now = time.monotonic()
        if progress_callback and (force or now - last_notice >= 2.0):
            progress_callback(phase, ep, epochs, batch, batches, score)
            last_notice = now
    for ep in range(1, epochs + 1):
        model.train()
        notice('training', ep, 0, len(loader), force=True)
        for batch, (x, y) in enumerate(loader, 1):
            x = (x + torch.randn_like(x) * 0.01).clamp(-2.1179, 2.64)
            x = _maybe_channels_last(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                _, loss = _bf16_forward_fp32_loss(model, x, y, criterion)
            else:
                loss = criterion(model(x).float(), y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss; no fabricated score')
            loss.backward()
            opt.step()
            notice('training', ep, batch, len(loader))
        model.eval()
        conf = np.zeros((10, 10), dtype=np.int64)
        sum_loss = 0.0
        count = 0
        val_bs = config['batch_size']
        vb = math.ceil(len(store.y) / val_bs)
        notice('validation', ep, 0, vb, force=True)
        with torch.inference_mode():
            for batch, (x, y) in enumerate(store.batches(val_bs, device), 1):
                x = _maybe_channels_last(x)
                if device.type == 'cuda':
                    logits, loss = _bf16_forward_fp32_loss(model, x, y, criterion)
                else:
                    logits = model(x)
                    loss = criterion(logits.float(), y)
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite validation; no fabricated score')
                truth = y.cpu().numpy()
                pred = logits.argmax(1).cpu().numpy()
                np.add.at(conf, (truth, pred), 1)
                sum_loss += float(loss) * len(y)
                count += len(y)
                notice('validation', ep, batch, vb)
        f1 = macro_f1(conf)
        vloss = sum_loss / count
        if best is None or (f1, -vloss) > (best[0], -best[1]):
            best = (f1, vloss, ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({'epoch': ep, 'val_macro_f1': f1, 'val_loss': vloss, 'learning_rate': opt.param_groups[0]['lr']})
        notice('epoch_done', ep, vb, vb, f1, force=True)
        schedule.step()
        if epoch_prune_callback and epoch_prune_callback(ep, f1, vloss):
            raise ValueError('This six-algorithm suite has no pruning')
    return {'model': model, 'best_state': best_state, 'best_val_macro_f1': best[0], 'best_val_loss': best[1], 'best_epoch': best[2], 'epochs_completed': epochs, 'pruned': False, 'numerical_failure': False, 'numerical_failure_reason': '', 'epoch_history': history, 'training_seconds': time.monotonic() - start, 'peak_vram_mb': torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else 0.0}
