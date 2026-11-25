from __future__ import annotations
"""
To Run:

python build_synthetic_dataset.py --src frigate/processed --out synthetic_dataset --n-train 20000 --n-val 2000 --n-test 2000
"""
"""
Build (and load) an on-disk synthetic moving-object dataset.

    python build_synthetic_dataset.py --src path/to/frigate/processed --out synthetic_dataset \
        --n-train 20000 --n-val 2000 --n-test 2000 --seed 0

Output folder layout
    <out>/
      manifest.json        config, seed, source frames, pairs, split definition, counts
      README.md            format description
      preview.png          a few positive samples (A | B | diff | mask overlay)
      train/ val/ test/
        images.npy         uint8 (N, 2, P, P)       [:,0]=frame A, [:,1]=frame B
        masks.npy          uint8 (N, 2, P, ceil(P/8))  bit-packed; unpack with unpack_mask(m, P)
        meta.jsonl         one JSON line per sample (objects, distractors, origin, pair)

Splits are by x-strip of the aligned frame (see STRIP_W / SPLIT_STRIPS): the frames
all show nearly the same star field and the field drifts mainly in y, so column
strips keep train/val/test star content (approximately) disjoint. Splitting by
frame would NOT, because every frame contains the same stars.

Samples are stored WITHOUT augmentation, so meta coordinates are exact; flips /
rot90 are applied at load time (StoredDataset(augment=True)).
"""


import argparse
import glob
import json
import os
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

import synthetic_moving_objects as S

STRIP_W = 384                                
SPLIT_STRIPS = {"train": [0, 1, 3, 5], "val": [2], "test": [4]}
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def pick_region(rng, split, width, patch):
    """Choose one strip of this split and return its (xmin, xmax) column range."""
    strips = [k for k in SPLIT_STRIPS[split] if min((k + 1) * STRIP_W, width) - k * STRIP_W >= patch]
    k = strips[int(rng.integers(len(strips)))]
    return k * STRIP_W, min((k + 1) * STRIP_W, width)


def build_split(pairs, cfg, split, n, seed, out_dir):
    d = os.path.join(out_dir, split)
    os.makedirs(d, exist_ok=True)
    P = cfg.patch
    images = np.lib.format.open_memmap(os.path.join(d, "images.npy"), "w+", np.uint8, (n, 2, P, P))
    masks = np.lib.format.open_memmap(os.path.join(d, "masks.npy"), "w+", np.uint8, (n, 2, P, (P + 7) // 8))
    with open(os.path.join(d, "meta.jsonl"), "w") as f:
        for i in range(n):
            rng = np.random.default_rng([seed, SPLIT_CODE[split], i]) 
            k = int(rng.integers(len(pairs)))
            A, B = pairs.get(k)
            xr = pick_region(rng, split, A.shape[1], P)
            s = S.make_sample(A, B, cfg, rng, x_range=xr)
            images[i, 0], images[i, 1] = s["A"], s["B"]
            masks[i] = np.packbits(s["mask"].astype(np.uint8), axis=-1)
            a, b = pairs.idx[k]
            f.write(json.dumps(dict(id=i, pair=[os.path.basename(pairs.paths[a]), os.path.basename(pairs.paths[b])],
                                    origin=list(s["origin"]), x_range=list(xr),
                                    objects=s["meta"], distractors=s["distractors"])) + "\n")
    images.flush(); masks.flush()


def unpack_mask(packed, width):
    """(..., P, ceil(P/8)) uint8 -> (..., P, width) float32; `width` = P drops the padding bits."""
    return np.unpackbits(packed, axis=-1, count=width).astype(np.float32)


class StoredDataset:
    """Reads a split folder written by build_split (duck-typed map-style dataset).

    The memory-mapped arrays are opened lazily and dropped when pickled, so DataLoader
    workers (which pickle the dataset on Windows) never copy the data.
    Items: dict(x=(3,P,P), y=(2,P,P), idx=int[, meta=dict]); torch tensors if torch exists.
    with_meta=False is needed for DataLoader batching (meta dicts can't be collated).
    """

    def __init__(self, folder, augment=False, with_meta=True):
        self.folder, self.augment, self.with_meta = folder, augment, with_meta
        self._images = self._masks = None
        with open(os.path.join(folder, "meta.jsonl")) as f:
            self.meta = [json.loads(line) for line in f]
        n_img = np.load(os.path.join(folder, "images.npy"), mmap_mode="r").shape[0]
        self._n = min(n_img, len(self.meta))     

    @property
    def images(self):
        if self._images is None:
            self._images = np.load(os.path.join(self.folder, "images.npy"), mmap_mode="r")
        return self._images

    @property
    def masks(self):
        if self._masks is None:
            self._masks = np.load(os.path.join(self.folder, "masks.npy"), mmap_mode="r")
        return self._masks

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_images"] = d["_masks"] = None
        return d

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        A, B = np.asarray(self.images[i, 0]), np.asarray(self.images[i, 1])
        y = unpack_mask(np.asarray(self.masks[i]), self.images.shape[-1])
        if self.augment:
            rng = np.random.default_rng()
            k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
            f = lambda a: np.ascontiguousarray(np.rot90(a[..., ::-1] if flip else a, k, axes=(-2, -1)))
            A, B, y = f(A), f(B), f(y)
        x = S.to_input(A, B)
        if S.torch is not None:
            import torch
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        out = dict(x=x, y=y, idx=i)
        if self.with_meta:
            out["meta"] = self.meta[i]
        return out


def make_preview(out_dir, split="test", n=4):
    ds = StoredDataset(os.path.join(out_dir, split))
    rows = []
    for i in [j for j, m in enumerate(ds.meta) if m["objects"]][:n]:
        A, B = np.asarray(ds.images[i, 0]), np.asarray(ds.images[i, 1])
        y = unpack_mask(np.asarray(ds.masks[i]), ds.images.shape[-1])
        g = lambda a: np.stack([np.clip(a.astype(float) * 2, 0, 255).astype(np.uint8)] * 3, -1)
        d = np.clip(128 + 3 * (B.astype(float) - A.astype(float)), 0, 255).astype(np.uint8)
        ov = np.stack([B] * 3, -1).copy()
        ov[y[1] > 0], ov[y[0] > 0] = (255, 0, 0), (0, 255, 0)
        rows.append(np.hstack([g(A), g(B), np.stack([d] * 3, -1), ov]))
    Image.fromarray(np.vstack(rows)).save(os.path.join(out_dir, "preview.png"))


README = """# Synthetic moving-object dataset

Real consecutive Frigate frames (stars, noise, residual misalignment) with injected
moving objects (points / short streaks) displaced between frame A and frame B.

Per split folder: images.npy (N,2,P,P) uint8 [A,B]; masks.npy (N,2,P,ceil(P/8)) bit-packed
[object in A, object in B] (build_synthetic_dataset.unpack_mask(m, P)); meta.jsonl with the
injected objects (centres in A and B, length, angle, peak amplitude, displacement, ...),
the distractors (hot pixels / cosmic-ray tracks present in only one frame, label 0),
patch origin and source pair.

Samples are un-augmented; meta coordinates refer to the stored patch.
Load with StoredDataset(folder, augment=True) for training.
See manifest.json for the generator config, seed and source frames.
Real, hand-verified streaks are NOT in here: keep them as a separate test set.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with Capture_*.png frames")
    ap.add_argument("--out", default="synthetic_dataset")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--patch", type=int, default=256, help=f"patch size in px (must be <= {STRIP_W})")
    ap.add_argument("--no-number-check", action="store_true",
                    help="pair frames by position in the sorted list, ignoring gaps in the frame numbers")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    paths = sorted(glob.glob(os.path.join(src, "Capture_*.png")))
    print(f"source folder : {src}")
    print(f"frames found  : {len(paths)}")
    if not paths:
        raise SystemExit("ERROR: no files matching Capture_*.png in the source folder - check --src")
    pairs = S.FramePairs(paths, gap=a.gap, require_consecutive=not a.no_number_check)
    print(f"usable pairs  : {len(pairs)}  (frame numbers must differ by exactly --gap={a.gap})")
    if len(pairs) == 0:
        raise SystemExit("ERROR: frames were found but none form a consecutive pair. "
                         "Check the numbering, or re-run with --no-number-check.")
    if a.patch > STRIP_W:
        raise SystemExit(f"ERROR: --patch {a.patch} is larger than the split strip width ({STRIP_W})")
    cfg = S.GenConfig(augment=False, patch=a.patch)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    print(f"writing to    : {out}")

    counts = dict(train=a.n_train, val=a.n_val, test=a.n_test)
    t0 = time.time()
    for split, n in counts.items():
        build_split(pairs, cfg, split, n, a.seed, out)
        print(f"  {split}: {n} samples  ({time.time() - t0:.0f}s)")

    manifest = dict(seed=a.seed, gap=a.gap, counts=counts, config=asdict(cfg),
                    strip_width=STRIP_W, split_strips=SPLIT_STRIPS,
                    frames=[os.path.basename(p) for p in paths],
                    pairs=[[os.path.basename(pairs.paths[i]), os.path.basename(pairs.paths[j])] for i, j in pairs.idx],
                    integer_shifts={f"{os.path.basename(pairs.paths[i])}->{os.path.basename(pairs.paths[j])}": list(pairs._shift.get(k, (0, 0)))
                                    for k, (i, j) in enumerate(pairs.idx)})
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README)
    make_preview(out)
    print("done ->", out)


if __name__ == "__main__":
    main()
