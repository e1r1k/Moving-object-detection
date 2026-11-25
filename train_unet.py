"""
Train a U-Net to find moving objects in (frame A, frame B) patches.
 
    python train_unet.py --data synthetic_dataset --out runs/unet1 --epochs 15
    python train_unet.py --data synthetic_dataset --out runs/unet1 --test-only     # re-evaluate best.pt
 
Input  : (3, P, P)  asinh z-scores of [A, B, B-A]          (from StoredDataset / to_input)
Target : (2, P, P)  mask of the object in A and in B
Output : (2, P, P)  logits; sigmoid -> probability map. "Contains a streak" is derived
         from it (any predicted blob >= --min-size px), there is no separate head.
 
Writes to --out:  config.json, history.json, last.pt, best.pt (best val pixel-F1),
                  test_report.json (threshold sweep + recall by brightness) after training.
 
Important: validation and test data come from the SAME generator as training, so a
good score here means "learned the synthetic task". It does not by itself show the
model works on real frames - keep real, hand-verified streaks as a separate test.
"""
from __future__ import annotations
 
import argparse
import json
import math
import os
import time
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader
 
from build_synthetic_dataset import StoredDataset
 
AMP_BINS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]   # peak counts above background (noise sigma ~6)
 
 
# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
 
 
class UNet(nn.Module):
    """Plain U-Net. `depth` 2x2 poolings -> input size must be divisible by 2**depth."""
 
    def __init__(self, in_ch=3, out_ch=2, base=32, depth=4):
        super().__init__()
        self.depth = depth
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([conv_block(in_ch, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)
        nn.init.constant_(self.head.bias, -4.6)   # start at p~0.01: objects are a tiny fraction of pixels
 
    def forward(self, x):
        m = 2 ** self.depth
        if x.shape[-1] % m or x.shape[-2] % m:
            raise ValueError(f"patch size must be divisible by {m}, got {tuple(x.shape[-2:])}")
        feats = [self.enc[0](x)]
        for e in self.enc[1:]:
            feats.append(e(self.pool(feats[-1])))
        x = feats.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([feats.pop(), up(x)], 1))
        return self.head(x)
 
 
# --------------------------------------------------------------------------
# Loss: focal (hard-pixel focus) + batch-level soft Dice (overlap)
# --------------------------------------------------------------------------
def focal_dice_loss(logits, target, gamma=2.0):
    logits = logits.float()
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    focal = ((1 - pt) ** gamma * bce).sum() / target.sum().clamp(min=1.0)
    inter = (p * target).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + target.sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return focal + dice, focal.detach(), dice.detach()
 
 
# --------------------------------------------------------------------------
# Evaluation (pixel, object, sample level)
# --------------------------------------------------------------------------
def _new_acc():
    return dict(tp=0, fp=0, fn=0, objs=[], fp_comps=0, n=0, n_neg=0, fp_comps_neg=0,
                s_tp=0, s_fp=0, s_fn=0, s_tn=0)
 
 
def _accumulate(acc, pred, gt, metas, min_size, win=4, near_iter=4):
    """pred, gt: bool (B,2,P,P); metas: list of per-sample meta dicts."""
    acc["tp"] += int((pred & gt).sum()); acc["fp"] += int((pred & ~gt).sum()); acc["fn"] += int((~pred & gt).sum())
    P = pred.shape[-1]
    for b, m in enumerate(metas):
        has_pred, fps = False, 0
        for c in (0, 1):
            lab, n = ndi.label(pred[b, c], structure=np.ones((3, 3)))
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            touched = set()
            if gt[b, c].any():
                near = ndi.binary_dilation(gt[b, c], iterations=near_iter)
                touched = set(np.unique(lab[near]).tolist())
            for k in range(1, n + 1):
                if sizes[k] >= min_size:
                    has_pred = True
                    fps += k not in touched
        acc["fp_comps"] += fps; acc["n"] += 1
        has_gt = len(m["objects"]) > 0
        if not has_gt:
            acc["n_neg"] += 1; acc["fp_comps_neg"] += fps
        acc["s_tp"] += has_pred and has_gt; acc["s_fp"] += has_pred and not has_gt
        acc["s_fn"] += (not has_pred) and has_gt; acc["s_tn"] += (not has_pred) and not has_gt
        for o in m["objects"]:            
            det = []
            for c, key in ((0, "cA"), (1, "cB")):
                cx, cy = int(round(o[key][0])), int(round(o[key][1]))
                det.append(bool(pred[b, c, max(0, cy - win):min(P, cy + win + 1), max(0, cx - win):min(P, cx + win + 1)].any()))
            acc["objs"].append((o["amp"], det[0], det[1]))
 
 
def _summarise(acc):
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    objs = acc["objs"]
    r = dict(pixel_precision=tp / max(tp + fp, 1), pixel_recall=tp / max(tp + fn, 1),
             pixel_f1=2 * tp / max(2 * tp + fp + fn, 1), pixel_iou=tp / max(tp + fp + fn, 1),
             obj_recall_B=float(np.mean([o[2] for o in objs])) if objs else float("nan"),
             obj_recall_A=float(np.mean([o[1] for o in objs])) if objs else float("nan"),
             obj_recall_both=float(np.mean([o[1] and o[2] for o in objs])) if objs else float("nan"),
             fp_blobs_per_image=acc["fp_comps"] / max(acc["n"], 1),
             fp_blobs_per_empty_image=acc["fp_comps_neg"] / max(acc["n_neg"], 1),
             sample_recall=acc["s_tp"] / max(acc["s_tp"] + acc["s_fn"], 1),
             sample_precision=acc["s_tp"] / max(acc["s_tp"] + acc["s_fp"], 1),
             sample_false_alarm_rate=acc["s_fp"] / max(acc["s_fp"] + acc["s_tn"], 1),
             n_samples=acc["n"], n_objects=len(objs))
    r["recall_B_by_amp"] = {}
    for lo, hi in AMP_BINS:
        sel = [o[2] for o in objs if lo <= o[0] < hi]
        r["recall_B_by_amp"][f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"] = (float(np.mean(sel)) if sel else None, len(sel))
    return r
 
 
@torch.no_grad()
def evaluate(model, ds, device, thresholds=(0.5,), batch_size=32, workers=0, min_size=2, max_samples=None):
    """Returns {threshold: metrics}. `ds` must be a StoredDataset with with_meta=False, augment=False."""
    model.eval()
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    loader = DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=batch_size, shuffle=False, num_workers=workers)
    accs = {t: _new_acc() for t in thresholds}
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x).float()).cpu().numpy()
        gt = batch["y"].numpy() > 0.5
        metas = [ds.meta[int(i)] for i in batch["idx"]]
        for t in thresholds:
            _accumulate(accs[t], prob > t, gt, metas, min_size)
    return {t: _summarise(a) for t, a in accs.items()}
 
 
def print_report(res):
    for t, r in res.items():
        print(f"  thr {t:.2f} | pixel P {r['pixel_precision']:.3f} R {r['pixel_recall']:.3f} F1 {r['pixel_f1']:.3f} | "
              f"object recall B {r['obj_recall_B']:.3f} (A&B {r['obj_recall_both']:.3f}) | "
              f"false blobs/image {r['fp_blobs_per_image']:.3f} | sample recall {r['sample_recall']:.3f} "
              f"false-alarm rate {r['sample_false_alarm_rate']:.3f}")
 
 
# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def lr_at(step, total, base_lr, warmup=300, min_frac=0.01):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
 
 
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root containing train/ val/ test/")
    ap.add_argument("--out", default="runs/unet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net base width (16 = ~4x faster on CPU)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (use 0 if you see Windows multiprocessing trouble)")
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for per-epoch validation")
    ap.add_argument("--min-size", type=int, default=2, help="min blob size (px) to count as a detection")
    ap.add_argument("--max-train-samples", type=int, default=None, help="use only the first N training samples (quick tests)")
    ap.add_argument("--val-samples", type=int, default=1000, help="validation samples per epoch (test uses all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--test-only", action="store_true", help="skip training; evaluate <out>/best.pt on the test split")
    a = ap.parse_args()
 
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = pick_device()
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device: {device}  (mixed precision: {use_amp})")
 
    model = UNet(base=a.base, depth=a.depth).to(device)
    print(f"U-Net base={a.base} depth={a.depth}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    val_ds = StoredDataset(os.path.join(a.data, "val"), augment=False, with_meta=False)
    test_ds = StoredDataset(os.path.join(a.data, "test"), augment=False, with_meta=False)
 
    if not a.test_only:
        train_ds = StoredDataset(os.path.join(a.data, "train"), augment=True, with_meta=False)
        n_train = len(train_ds) if a.max_train_samples is None else min(a.max_train_samples, len(train_ds))
        loader = DataLoader(torch.utils.data.Subset(train_ds, range(n_train)), batch_size=a.batch_size, shuffle=True,
                            num_workers=a.workers, drop_last=True, pin_memory=device.type == "cuda",
                            persistent_workers=a.workers > 0)
        steps_per_epoch = len(loader)
        total = steps_per_epoch * a.epochs
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        scaler = make_scaler(use_amp)
        history, start_epoch, best, step = [], 0, -1.0, 0
        ck = os.path.join(a.out, "last.pt")
        if a.resume and os.path.exists(ck):
            s = torch.load(ck, map_location=device, weights_only=False)   # our own file
            model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
            history, start_epoch, best = s["history"], s["epoch"] + 1, s["best"]
            step = start_epoch * steps_per_epoch
            print(f"resumed from epoch {start_epoch}")
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(vars(a) | dict(n_train=n_train, steps_per_epoch=steps_per_epoch), f, indent=2)
        print(f"training on {n_train} samples, {steps_per_epoch} steps/epoch, {a.epochs} epochs")
 
        for epoch in range(start_epoch, a.epochs):
            model.train(); t0 = time.time(); run = np.zeros(3); nb = 0
            for batch in loader:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total, a.lr)
                x, y = batch["x"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x)
                else:
                    logits = model(x)
                loss, fl, dc = focal_dice_loss(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                run += [loss.item(), fl.item(), dc.item()]; nb += 1; step += 1
                if nb % 100 == 0:
                    print(f"  epoch {epoch + 1} step {nb}/{steps_per_epoch}  loss {run[0] / nb:.4f}", flush=True)
            res = evaluate(model, val_ds, device, thresholds=(a.thr,), workers=0, min_size=a.min_size,
                           max_samples=a.val_samples)[a.thr]
            rec = dict(epoch=epoch + 1, train_loss=float(run[0] / nb), train_focal=float(run[1] / nb), train_dice=float(run[2] / nb),
                       lr=float(opt.param_groups[0]["lr"]), seconds=float(time.time() - t0),
                       **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in res.items() if k != "recall_B_by_amp"})
            history.append(rec)
            print(f"epoch {epoch + 1}/{a.epochs}  {rec['seconds']:.0f}s  loss {rec['train_loss']:.4f} | val F1 {res['pixel_f1']:.3f} "
                  f"obj recall {res['obj_recall_B']:.3f} false blobs/img {res['fp_blobs_per_image']:.3f}", flush=True)
            state = dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, best=best, history=history, args=vars(a))
            if res["pixel_f1"] > best:
                best = state["best"] = res["pixel_f1"]
                torch.save(state, os.path.join(a.out, "best.pt"))
            torch.save(state, ck)
            with open(os.path.join(a.out, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
 

    # FINAL TEST EVALUATION
    best_path = os.path.join(a.out, "best.pt")
    if not os.path.exists(best_path):
        raise SystemExit(f"no checkpoint at {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    print("\nTEST (best checkpoint), threshold sweep:")
    ths = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    res = evaluate(model, test_ds, device, thresholds=ths, workers=0, min_size=a.min_size)
    print_report(res)
    r = res[0.5]
    print("\nobject recall (frame B) by peak brightness at thr 0.5   [noise sigma ~ 6 levels]")
    for k, (v, n) in r["recall_B_by_amp"].items():
        print(f"  peak {k:>8s}: {('%.3f' % v) if v is not None else '  n/a'}   (n={n})")
    with open(os.path.join(a.out, "test_report.json"), "w") as f:
        json.dump({str(t): v for t, v in res.items()}, f, indent=2)
    print(f"\nreport saved to {os.path.join(a.out, 'test_report.json')}")
 
 
if __name__ == "__main__":
    main()