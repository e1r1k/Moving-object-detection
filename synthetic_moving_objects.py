from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.special import erf
from skimage.registration import phase_cross_correlation

try:  
    import torch
    from torch.utils.data import Dataset as _Base
except ImportError:  
    torch = None
    _Base = object


    
@dataclass
class GenConfig:
    patch: int = 265
    border: int = 16

    n_objects_probs: tuple = (0.2, 0.6, 0.2)
    amp_range: tuple = (6, 150)
    sigma_range: tuple = (0.5, 0.75)
    point_prob: float = 0.15
    length_range: tuple = (2, 20)
    disp_range: tuple = (4, 80)    
    flicker_max : float = 0.5
    frame_gain_range: tuple = (0.7,  1.3)
    gain: float = 2

    p_distractor: float = 0.5
    max_distractors: int = 3
    
    mask_frac: float = 0.25
    augment: bool = True


def render_streak(shape, center, length, angle, sigma, amp, flicker=0, phase=0, cycles=1):

    H, W = shape
    cx, cy = center

    n = int(np.ceil(length*4)) + 1 if length > 0 else 1
    t = np.linspace(-0.5, 0.5, n) if n > 1 else np.zeros(1)

    px = cx + t * length * np.cos(angle)
    py = cy + t * length * np.sin(angle)
    pad = int(np.ceil(4*sigma))+1
    x0, x1 = max(0, int(np.floor(px.min())) - pad), min(W, int(np.ceil(px.max())) + pad + 1)
    y0, y1 = max(0, int(np.floor(py.min())) - pad), min(H, int(np.ceil(py.max())) + pad + 1)
    out = np.zeros(shape, np.float32)

    if x1 <= x0 or y1 <= y0:
        return out
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    s2 = sigma * np.sqrt(2)
    wx = 0.5 * (erf((xs[None] + 0.5 - px[:, None]) / s2) - erf((xs[None] - 0.5 - px[:, None]) / s2))
    wy = 0.5 * (erf((ys[None] + 0.5 - py[:, None]) / s2) - erf((ys[None] - 0.5 - py[:, None]) / s2))
    w = 1 + flicker * np.sin(2*np.pi*cycles*(t+0.5)+phase)
    img = (wy * w[:, None]).T @ wx
    if img.max() > 0:
        out[y0:y1, x0:x1] = img * (amp / img.max())
    return out

def add_signal_noise(base, signal, gain, rng):
    noise = rng.normal(size=signal.shape).astype(np.float32) * np.sqrt(gain * np.clip(signal, 0, None))
    return base + signal + noise

def _frame_number(path):
    m = re.search(r"Capture_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def load_frame(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a

class FramePairs:

    def __init__(self, paths, gap=1, border=16, align=True, require_consecutive=True):
        self.paths, self.gap, self.border, self.align = sorted(paths), gap, border, align
        nums = [_frame_number(p) for p in self.paths]
        use_nums = require_consecutive and all(n is not None for n in nums)
        self.idx = [(i, i + gap) for i in range(len(self.paths) - gap)
                    if not use_nums or nums[i + gap] - nums[i] == gap]
        self._shift = {}

    def __len__(self):
        return len(self.idx)

    @staticmethod
    @lru_cache(maxsize=16)
    def _load(path):
        return load_frame(path)

    def get(self, k):
        i, j = self.idx[k]
        A, B = self._load(self.paths[i]), self._load(self.paths[j])
        H, W = A.shape
        b = self.border
        if self.align:
            if k not in self._shift:
                cy, cx = H // 2, W // 2
                sl = (slice(cy - 512, cy + 512), slice(cx - 512, cx + 512))
                s, _, _ = phase_cross_correlation(A[sl].astype(np.float32), B[sl].astype(np.float32))
                self._shift[k] = (int(round(s[0])), int(round(s[1])))
            dy, dx = self._shift[k]
        else:
            dy = dx = 0
        ya, yb = max(0, dy) + b, min(H, H + dy) - b
        xa, xb = max(0, dx) + b, min(W, W + dx) - b
        return A[ya:yb, xa:xb], B[ya-dy:yb-dy, xa-dx:xb-dx]


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _sample_object(rng, cfg, P):
    sigma = rng.uniform(*cfg.sigma_range)
    length = 0 if rng.random() < cfg.point_prob else rng.uniform(*cfg.length_range)
    angle = rng.uniform(0, 2 * np.pi)
    margin = int(np.ceil(4 * sigma)) + 2 + length / 2
    D = _loguniform(rng, *cfg.disp_range)
    D = min(D, P - 2 * margin - 2)
    dx, dy = D * np.cos(angle), D * np.sin(angle)
    lo_x, hi_x = max(margin, margin-dx), min(P - margin, P - margin - dx)
    lo_y, hi_y = max(margin, margin-dy), min(P - margin, P - margin - dy)

    cx, cy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    amp = _loguniform(rng, *cfg.amp_range)

    return dict(cA=(cx,cy),
                cB = (cx + dx, cy + dy),
                length = length,
                angle = angle,
                sigma = sigma,
                amp = amp,
                disp = D,
                flicker = rng.uniform(0, cfg.flicker_max),
                phase = rng.uniform(0, 2 * np.pi), 
                cycles = rng.uniform(0.5, 2),
                gain_B = rng.uniform(*cfg.frame_gain_range))

def make_sample(A_full, B_full, cfg: GenConfig, rng: np.random.Generator, x_range=None):

    P = cfg.patch
    H, W = A_full.shape
    xmin, xmax = (0, W) if x_range is None else x_range
    y0, x0 = int(rng.integers(0, H - P + 1)), int(rng.integers(xmin, xmax - P + 1))
    A = A_full[y0:y0 + P, x0:x0 + P].astype(np.float32)
    B = B_full[y0:y0 + P, x0:x0 + P].astype(np.float32)
    mask = np.zeros((2, P, P), np.float32)
    meta, distractors = [], []
 
    n_obj = rng.choice(len(cfg.n_objects_probs), p=cfg.n_objects_probs)
    for _ in range(n_obj):
        o = _sample_object(rng, cfg, P)
        sA = render_streak((P, P), o["cA"], o["length"], o["angle"], o["sigma"], o["amp"],
                           o["flicker"], o["phase"], o["cycles"])
        sB = render_streak((P, P), o["cB"], o["length"], o["angle"], o["sigma"], o["amp"] * o["gain_B"],
                           o["flicker"], o["phase"], o["cycles"])
        A = add_signal_noise(A, sA, cfg.gain, rng)
        B = add_signal_noise(B, sB, cfg.gain, rng)
        mask[0] = np.maximum(mask[0], sA > cfg.mask_frac * sA.max())
        mask[1] = np.maximum(mask[1], sB > cfg.mask_frac * sB.max())
        meta.append({k: (v if not isinstance(v, tuple) else tuple(map(float, v))) for k, v in o.items()})

 
    if rng.random() < cfg.p_distractor:
        for _ in range(rng.integers(1, cfg.max_distractors + 1)):
            hot = rng.random() < 0.5                       
            s = render_streak((P, P), (rng.uniform(8, P - 8), rng.uniform(8, P - 8)),
                              0.0 if hot else rng.uniform(2, 10), rng.uniform(0, 2 * np.pi),
                              0.4, rng.uniform(30, 200))
            in_A = rng.random() < 0.5
            if in_A:
                A = add_signal_noise(A, s, cfg.gain, rng)
            else:
                B = add_signal_noise(B, s, cfg.gain, rng)
            distractors.append(dict(frame="A" if in_A else "B", kind="hot_pixel" if hot else "cosmic_ray",
                                    peak=float(s.max())))
 
    A = np.clip(np.rint(A), 0, 255).astype(np.uint8)   
    B = np.clip(np.rint(B), 0, 255).astype(np.uint8)
 
    if cfg.augment:
        k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        f = lambda a: np.ascontiguousarray(np.rot90(a[..., ::-1] if flip else a, k, axes=(-2, -1)))
        A, B, mask = f(A), f(B), f(mask)
        # NOTE: meta coordinates refer to the un-augmented patch.
    return dict(A=A, B=B, mask=mask, meta=meta, distractors=distractors, origin=(y0, x0))
 

def to_input(A, B, sigma=6, knee=4):
    A, B = A.astype(np.float32), B.astype(np.float32)
    za, zb = (A - np.median(A)) / sigma, (B - np.median(B)) / sigma
    return np.arcsinh(np.stack([za, zb, zb-za]) / knee).astype(np.float32)

class SyntheticDataset(_Base):
    def __init__(self, 
                 pairs: FramePairs, 
                 cfg: GenConfig = GenConfig(), 
                 length=10000, 
                 seed=0, 
                 deterministic=False):
        self.pairs, self.cfg, self.length = pairs, cfg, length
        self.seed, self.deterministic = seed, deterministic

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i]) if self.deterministic else np.random.default_rng()
        A, B = self.pairs.get(int(rng.integers(len(self.pairs))))
        s = make_sample(A, B, self.cfg, rng)
        x, y = to_input(s["A"], s["B"]), s["mask"]
        if torch is None:
            return dict(x=x, y=y, meta=s["meta"])
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), meta=s["meta"])



if __name__ == "__main__":
    import glob, os
    paths = sorted(glob.glob("frigate/processed/*.png"))
    print(paths)
    print(os.getcwd(), len(paths), paths[:2])
    assert len(paths) >= 2, "need at least 2 frames to form a pair"
    pairs = FramePairs(paths)

    cfg = GenConfig(augment=False, n_objects_probs=(0, 1, 0), p_distractor=0)
    rng = np.random.default_rng(9)

    rows = []
    for _ in range(2):
        A, B = pairs.get(int(rng.integers(len(pairs))))
        s = make_sample(A, B, cfg, rng)
        d = np.clip(128 + 3 * (s["B"].astype(float) - s["A"].astype(float)), 0, 255)
        ov = np.stack([s["B"]] * 3, -1).astype(np.uint8)
        ov[s["mask"][1] > 0] = (255, 0, 0)
        ov[s["mask"][0] > 0] = (0, 255, 0)
        rows.append(np.hstack([np.stack([np.clip(s["A"] * 2, 0, 255).astype(np.uint8)] * 3, -1),
                                        np.stack([np.clip(s["B"] * 2, 0, 255).astype(np.uint8)] * 3, -1),
                                        np.stack([d.astype(np.uint8)] * 3, -1), ov]))
    Image.fromarray(np.vstack(rows)).save("generator_demo.png")