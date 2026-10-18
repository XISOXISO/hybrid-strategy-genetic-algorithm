from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import hashlib, importlib.resources, math, sys, types
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
D30_EXT_DIR = SCRIPT_DIR
D30_EXTENSION_SHA256 = {'F1': '41cd4647f35430d3be0268189d61723bacda5ed1d4978cd3e3d6cd63249512c8', 'F2': 'b14c5ce723cb7aed7df91ba7a2f1250f2dd6c427563162f9c1f741b37bd860bb', 'F3': '3ca3fa15404ce18ad5f02fdf9f1e096d82180704620edfa8d25586c32c67e0b2', 'F4': '1fbb9d993e80c24d4077d589a13c3fd5091d95b606f2fda17abe0919e86736ea', 'F5': 'a07a2a0fc26fd39dd19e7660d4b286d1f4898fdb938fde1e07dcdc1632db19c8', 'F6': '234bdc8f6b0b137323c0bd8bec27a1d85242105f58f5b8b4615c92a3b469e921', 'F7': 'c6f95190b287fa5c1faf77bb8c2f301f9775509313d8a4cb0719194d450dec76', 'F8': '6bedc086193715bc5533a5768a4e38966033501bf75ade2aa8277a061103d4c1', 'F9': '2115befdc6132e74e5ef85d5f69efc659ee070d9eb46b2595c2b4566fb2bf3e6', 'F10': '8d97145348db6d9d6835530332fa1f9381cb964e05967a23c8bb2da353957e82', 'F11': '6439dddaed201b4526ea7e51f26e0f146dac5b91aea3fe55deeb919641d3f226', 'F12': '8ff305dfa7f6ddcdf8eab0428850b715fc693212e8835a32e4c735701458970d'}
OFFICIAL_BIASES = {1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1800.0, 7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0}

def _ensure_pkg_resources_compatibility() -> None:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)
            import pkg_resources
        return
    except Exception:
        pass
    m = types.ModuleType('pkg_resources')

    def resource_filename(package_name: str, relative_path: str) -> str:
        return str(importlib.resources.files(package_name).joinpath(relative_path))
    m.resource_filename = resource_filename
    sys.modules['pkg_resources'] = m

def import_cec_classes() -> Dict[int, Any]:
    _ensure_pkg_resources_compatibility()
    from opfunu.cec_based.cec2022 import F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, F122022
    return {1: F12022, 2: F22022, 3: F32022, 4: F42022, 5: F52022, 6: F62022, 7: F72022, 8: F82022, 9: F92022, 10: F102022, 11: F112022, 12: F122022}

def instantiate_opfunu(cls: Any, dim: int) -> Any:
    for kwargs in ({'ndim': dim}, {'dim': dim}):
        try:
            return cls(**kwargs)
        except Exception:
            pass
    raise RuntimeError(f'Cannot instantiate {cls} at D{dim}')

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def _refresh_hybrid_partitions(name: str, f: Any) -> None:
    d = int(f.ndim)
    if name == 'F6':
        f.n1 = int(np.ceil(f.p[0] * d))
        f.n2 = int(np.ceil(f.p[1] * d)) + f.n1
        f.idx1 = f.f_shuffle[:f.n1]
        f.idx2 = f.f_shuffle[f.n1:f.n2]
        f.idx3 = f.f_shuffle[f.n2:d]
    elif name == 'F7':
        f.n1 = int(np.ceil(f.p[0] * d))
        f.n2 = int(np.ceil(f.p[1] * d)) + f.n1
        f.n3 = int(np.ceil(f.p[2] * d)) + f.n2
        f.n4 = int(np.ceil(f.p[3] * d)) + f.n3
        f.n5 = int(np.ceil(f.p[4] * d)) + f.n4
        f.idx1 = f.f_shuffle[:f.n1]
        f.idx2 = f.f_shuffle[f.n1:f.n2]
        f.idx3 = f.f_shuffle[f.n2:f.n3]
        f.idx4 = f.f_shuffle[f.n3:f.n4]
        f.idx5 = f.f_shuffle[f.n4:f.n5]
        f.idx6 = f.f_shuffle[f.n5:d]
    elif name == 'F8':
        f.n1 = int(np.ceil(f.p[0] * d))
        f.n2 = int(np.ceil(f.p[1] * d)) + f.n1
        f.n3 = int(np.ceil(f.p[2] * d)) + f.n2
        f.n4 = int(np.ceil(f.p[3] * d)) + f.n3
        f.idx1 = f.f_shuffle[:f.n1]
        f.idx2 = f.f_shuffle[f.n1:f.n2]
        f.idx3 = f.f_shuffle[f.n2:f.n3]
        f.idx4 = f.f_shuffle[f.n3:f.n4]
        f.idx5 = f.f_shuffle[f.n4:d]

def instantiate_d30_ext(fid: int, classes: Dict[int, Any]) -> Any:
    name = f'F{fid}'
    path = D30_EXT_DIR / f'cec2022_{name}_D30_EXT.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    exp = D30_EXTENSION_SHA256[name]
    if got != exp:
        raise RuntimeError(f'{name} D30_EXT checksum mismatch')
    with np.load(path, allow_pickle=False) as z:
        shift = np.asarray(z['shift'], dtype=np.float64).copy()
        rot = np.asarray(z['rotation'], dtype=np.float64).copy()
        shuffle = np.asarray(z['shuffle'], dtype=int).copy() if 'shuffle' in z.files else None
    f = instantiate_opfunu(classes[fid], 20)
    f._ndim = 30
    f._bounds = np.asarray([[-100.0, 100.0]] * 30, dtype=np.float64)
    f.dim_default = 30
    f.dim_max = 30
    f.dim_supported = [30]
    f.f_shift = shift
    f.f_matrix = rot
    if name in {'F6', 'F7', 'F8'}:
        if shuffle is None:
            raise RuntimeError(f'missing shuffle for {name}')
        f.f_shuffle = shuffle
        _refresh_hybrid_partitions(name, f)
    f.x_global = f.f_shift.copy() if f.f_shift.ndim == 1 else f.f_shift[0].copy()
    f.n_fe = 0
    return f

def make_cpu_function(fid: int, dim: int, classes: Dict[int, Any]) -> Any:
    return instantiate_d30_ext(fid, classes) if dim == 30 else instantiate_opfunu(classes[fid], dim)

def torch_import():
    import torch
    return torch

def _rounder_t(x, condition):
    t = 2 * x
    inter = torch_import().trunc(t)
    dec = t - inter
    t = torch_import().where(t <= 0.0, inter - (dec >= 0.5).to(x.dtype), t)
    t = torch_import().where(dec < 0.5, inter, t)
    t = torch_import().where(dec >= 0.5, inter + 1.0, t)
    return torch_import().where(condition < 0.5, x, t / 2.0)

def rastrigin(x):
    t = torch_import()
    return (x * x - 10 * t.cos(2 * math.pi * x) + 10).sum(dim=1)

def rosenbrock(x):
    return (100 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1) ** 2).sum(dim=1)

def zakharov(x):
    s = (0.5 * x).sum(dim=1)
    return (x * x).sum(dim=1) + s * s + s ** 4

def rotated_schaffer(x):
    t = torch_import()
    y = t.roll(x, -1, dims=1)
    ss = x * x + y * y
    return (0.5 + (t.sin(t.sqrt(ss)) ** 2 - 0.5) / (1 + 0.001 * ss) ** 2).sum(dim=1)

def noncont_rastrigin(x):
    t = torch_import()
    y = _rounder_t(x, t.abs(x))
    yr = t.roll(y, -1, dims=1)
    return rastrigin(y) + rastrigin(yr)

def levy(x, shift=0.0):
    t = torch_import()
    x = x + shift
    w = 1 + (x - 1) / 4
    a = t.sin(math.pi * w[:, 0]) ** 2 + (w[:, -1] - 1) ** 2 * (1 + t.sin(2 * math.pi * w[:, -1]) ** 2)
    b = ((w[:, :-1] - 1) ** 2 * (1 + 10 * t.sin(math.pi * w[:, :-1] + 1) ** 2)).sum(dim=1)
    return a + b

def bent_cigar(x):
    return x[:, 0] ** 2 + 1000000.0 * (x[:, 1:] ** 2).sum(dim=1)

def discus(x):
    return 1000000.0 * x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(dim=1)

def hgbat(x, shift=0.0):
    t = torch_import()
    z = x + shift
    n = z.shape[1]
    s = z.sum(dim=1)
    s2 = (z * z).sum(dim=1)
    return t.abs(s2 * s2 - s * s).sqrt() + (0.5 * s2 + s) / n + 0.5

def happy_cat(x, shift=0.0):
    t = torch_import()
    z = x + shift
    n = z.shape[1]
    s = z.sum(dim=1)
    s2 = (z * z).sum(dim=1)
    return t.abs(s2 - n).pow(0.25) + (0.5 * s2 + s) / n + 0.5

def ackley(x):
    t = torch_import()
    n = x.shape[1]
    s2 = (x * x).sum(dim=1)
    c = t.cos(2 * math.pi * x).sum(dim=1)
    return -20 * t.exp(-0.2 * t.sqrt(s2 / n)) - t.exp(c / n) + 20 + math.e

def griewank(x):
    t = torch_import()
    n = x.shape[1]
    idx = t.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return (x * x).sum(dim=1) / 4000 - t.cos(x / t.sqrt(idx)).prod(dim=1) + 1

def elliptic(x):
    t = torch_import()
    n = x.shape[1]
    if n <= 1:
        return (x * x).sum(dim=1)
    idx = t.arange(n, dtype=x.dtype, device=x.device)
    coeff = t.pow(t.tensor(10.0, dtype=x.dtype, device=x.device), 6 * idx / (n - 1))
    return (x * x * coeff).sum(dim=1)

def grie_rosen(x):
    t = torch_import()
    z = x + 1.0
    z2 = t.roll(z, -1, dims=1)
    temp = 100 * (z * z - z2) ** 2 + (z - 1) ** 2
    return (temp * temp / 4000 - t.cos(temp) + 1).sum(dim=1)

def modified_schwefel(x):
    t = torch_import()
    z = x + 420.9687462275036
    n = x.shape[1]
    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = t.zeros_like(z)
    a = t.fmod(z, 500)
    aa = t.fmod(t.abs(z), 500)
    v1 = (500 - a) * t.sin(t.sqrt(t.clamp(500 - a, min=0))) - ((z - 500) / 100) ** 2 / n
    v2 = (-500 + aa) * t.sin(t.sqrt(t.clamp(500 - aa, min=0))) - ((z + 500) / 100) ** 2 / n
    v3 = z * t.sin(t.sqrt(t.abs(z)))
    fx = t.where(mask1, -v1, fx)
    fx = t.where(mask2, -v2, fx)
    fx = t.where(mask3, -v3, fx)
    return fx.sum(dim=1) + 418.9828872724338 * n

def katsuura(x):
    t = torch_import()
    n = x.shape[1]
    result = t.ones(x.shape[0], dtype=x.dtype, device=x.device)
    js = t.arange(1, 33, dtype=x.dtype, device=x.device)
    powers = t.pow(t.tensor(2.0, dtype=x.dtype, device=x.device), js)
    for i in range(n):
        q = x[:, i:i + 1] * powers[None, :]
        temp = (t.abs(q - t.round(q)) / powers[None, :]).sum(dim=1)
        result = result * (1 + (i + 1) * temp) ** (10 / n ** 1.2)
    return (result - 1) * 10 / (n * n)

def schaffer_f7(x):
    t = torch_import()
    n = x.shape[1]
    if n < 2:
        return x[:, 0] ** 2
    tt = x[:, :-1] ** 2 + x[:, 1:] ** 2
    r = t.sqrt(tt) * (t.sin(50 * tt.pow(0.2)) + 1)
    return (r.sum(dim=1) / (n - 1)) ** 2

def calc_weight(dx, delta):
    t = torch_import()
    n = dx.shape[1]
    temp = (dx * dx).sum(dim=1)
    regular = t.sqrt(1.0 / temp.clamp_min(t.finfo(dx.dtype).tiny)) * t.exp(-temp / (2 * n * delta * delta))
    return t.where(temp != 0, regular, t.full_like(temp, 1e+99))

@dataclass
class Support:
    fid: int
    dim: int
    bias: float
    shift: Any
    matrix: Any
    shuffle: Optional[Any]

class TorchCEC2022:

    def __init__(self, fid: int, dim: int, cpu_func: Any, device: str):
        t = torch_import()
        self.t = t
        self.fid = fid
        self.dim = dim
        self.device = t.device(device)
        self.dtype = t.float64
        self.bias = float(cpu_func.f_bias)
        self.shift = t.as_tensor(np.asarray(cpu_func.f_shift, dtype=np.float64), dtype=self.dtype, device=self.device)
        self.matrix = t.as_tensor(np.asarray(cpu_func.f_matrix, dtype=np.float64), dtype=self.dtype, device=self.device)
        sh = getattr(cpu_func, 'f_shuffle', None)
        self.shuffle = t.as_tensor(np.asarray(sh, dtype=np.int64), dtype=t.long, device=self.device) if sh is not None else None
        self.cpu_func = cpu_func
        if fid == 6:
            self.p = [0.4, 0.4, 0.2]
        elif fid == 7:
            self.p = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
        elif fid == 8:
            self.p = [0.3, 0.2, 0.2, 0.1, 0.2]
        else:
            self.p = None

    def mat(self, x, m):
        return x @ m.T

    def evaluate(self, x):
        t = self.t
        if not t.is_tensor(x):
            x = t.as_tensor(x, dtype=self.dtype, device=self.device)
        x = x.to(device=self.device, dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        f = self.fid
        b = self.bias
        if f == 1:
            z = self.mat(x - self.shift, self.matrix)
            return zakharov(z) + b
        if f == 2:
            z = self.mat(2.048 * (x - self.shift) / 100, self.matrix) + 1
            return rosenbrock(z) + b
        if f == 3:
            z = self.mat(0.5 * (x - self.shift) / 100, self.matrix)
            return rotated_schaffer(z) + b
        if f == 4:
            z = self.mat(5.12 * (x - self.shift) / 100, self.matrix)
            return noncont_rastrigin(z) + b
        if f == 5:
            z = self.mat(5.12 * (x - self.shift) / 100, self.matrix)
            return levy(z, 1.0) + b
        if f in (6, 7, 8):
            z = x - self.shift
            idx = self.shuffle
            assert idx is not None
            z1 = z[:, idx]
            mz = self.mat(z1, self.matrix)
            d = self.dim
            cuts = []
            s = 0
            for pp in self.p[:-1]:
                s += int(math.ceil(pp * d))
                cuts.append(s)
            seg = []
            a = 0
            for c in cuts + [d]:
                seg.append(mz[:, a:c])
                a = c
            if f == 6:
                return bent_cigar(seg[0]) + hgbat(seg[1], -1.0) + rastrigin(seg[2]) + b
            if f == 7:
                return hgbat(seg[0], -1.0) + katsuura(seg[1]) + ackley(seg[2]) + rastrigin(seg[3]) + modified_schwefel(seg[4]) + schaffer_f7(seg[5]) + b
            return katsuura(seg[0]) + happy_cat(seg[1], -1.0) + grie_rosen(seg[2]) + modified_schwefel(seg[3]) + ackley(seg[4]) + b
        if f == 9:
            lam = [1, 1e-06, 1e-06, 1e-06, 1e-06]
            bb = [0, 200, 300, 100, 400]
            delta = [10, 20, 30, 40, 50]
            vals = []
            ws = []
            z = self.mat(2.048 * (x - self.shift[0]) / 100, self.matrix[:self.dim]) + 1
            vals.append(lam[0] * rosenbrock(z) + bb[0])
            ws.append(calc_weight(x - self.shift[0], delta[0]))
            z = self.mat(x - self.shift[0], self.matrix[self.dim:2 * self.dim])
            vals.append(lam[1] * elliptic(z) + bb[1])
            ws.append(calc_weight(x - self.shift[1], delta[1]))
            z = self.mat(x - self.shift[0], self.matrix[2 * self.dim:3 * self.dim])
            vals.append(lam[2] * bent_cigar(z) + bb[2])
            ws.append(calc_weight(x - self.shift[2], delta[2]))
            z = self.mat(x - self.shift[0], self.matrix[3 * self.dim:4 * self.dim])
            vals.append(lam[3] * discus(z) + bb[3])
            ws.append(calc_weight(x - self.shift[3], delta[3]))
            z = self.mat(x - self.shift[0], self.matrix[4 * self.dim:5 * self.dim])
            vals.append(lam[4] * elliptic(z) + bb[4])
            ws.append(calc_weight(x - self.shift[4], delta[4]))
        elif f == 10:
            lam = [1, 1, 1]
            bb = [0, 200, 100]
            delta = [20, 10, 10]
            vals = []
            ws = []
            z = self.mat(10 * (x - self.shift[0]), self.matrix[:self.dim])
            vals.append(modified_schwefel(z) + bb[0])
            ws.append(calc_weight(x - self.shift[0], delta[0]))
            z = self.mat(5.12 * (x - self.shift[0]) / 100, self.matrix[self.dim:2 * self.dim])
            vals.append(rastrigin(z) + bb[1])
            ws.append(calc_weight(x - self.shift[1], delta[1]))
            z = self.mat(5 * (x - self.shift[0]) / 100, self.matrix[2 * self.dim:3 * self.dim])
            vals.append(hgbat(z, -1.0) + bb[2])
            ws.append(calc_weight(x - self.shift[2], delta[2]))
        elif f == 11:
            lam = [1e-26, 10, 1e-06, 10, 0.0005]
            bb = [0, 200, 300, 400, 200]
            delta = [20, 20, 30, 30, 20]
            vals = []
            ws = []
            z = self.mat(0.5 * (x - self.shift[0]) / 100, self.matrix[:self.dim])
            vals.append(lam[0] * rotated_schaffer(z) + bb[0])
            ws.append(calc_weight(x - self.shift[0], delta[0]))
            z = self.mat(1000 * (x - self.shift[0]) / 100, self.matrix[self.dim:2 * self.dim])
            vals.append(lam[1] * modified_schwefel(z) + bb[1])
            ws.append(calc_weight(x - self.shift[1], delta[1]))
            z = self.mat(600 * (x - self.shift[0]) / 100, self.matrix[2 * self.dim:3 * self.dim])
            vals.append(lam[2] * griewank(z) + bb[2])
            ws.append(calc_weight(x - self.shift[2], delta[2]))
            z = self.mat(2.048 * (x - self.shift[0]) / 100, self.matrix[3 * self.dim:4 * self.dim])
            vals.append(lam[3] * rosenbrock(z) + bb[3])
            ws.append(calc_weight(x - self.shift[3], delta[3]))
            z = self.mat(x - self.shift[0], self.matrix[4 * self.dim:5 * self.dim])
            vals.append(lam[4] * rastrigin(z) + bb[4])
            ws.append(calc_weight(x - self.shift[4], delta[4]))
        elif f == 12:
            lam = [10, 10, 2.5, 1e-26, 1e-06, 0.0005]
            bb = [0, 300, 500, 100, 400, 200]
            delta = [10, 20, 30, 40, 50, 60]
            vals = []
            ws = []
            z = self.mat(5 * (x - self.shift[0]) / 100, self.matrix[:self.dim])
            vals.append(lam[0] * hgbat(z, -1.0) + bb[0])
            ws.append(calc_weight(x - self.shift[0], delta[0]))
            z = self.mat(5.12 * (x - self.shift[0]) / 100, self.matrix[self.dim:2 * self.dim])
            vals.append(lam[1] * rastrigin(z) + bb[1])
            ws.append(calc_weight(x - self.shift[1], delta[1]))
            z = self.mat(1000 * (x - self.shift[0]) / 100, self.matrix[2 * self.dim:3 * self.dim])
            vals.append(lam[2] * modified_schwefel(z) + bb[2])
            ws.append(calc_weight(x - self.shift[2], delta[2]))
            z = self.mat(x - self.shift[0], self.matrix[3 * self.dim:4 * self.dim])
            vals.append(lam[3] * bent_cigar(z) + bb[3])
            ws.append(calc_weight(x - self.shift[3], delta[3]))
            z = self.mat(x - self.shift[0], self.matrix[4 * self.dim:5 * self.dim])
            vals.append(lam[4] * elliptic(z) + bb[4])
            ws.append(calc_weight(x - self.shift[4], delta[4]))
            z = self.mat(x - self.shift[0], self.matrix[5 * self.dim:6 * self.dim])
            vals.append(lam[5] * rotated_schaffer(z) + bb[5])
            ws.append(calc_weight(x - self.shift[5], delta[5]))
        else:
            raise ValueError(f)
        W = t.stack(ws, dim=1)
        G = t.stack(vals, dim=1)
        W = W / W.sum(dim=1, keepdim=True)
        return (W * G).sum(dim=1) + b
