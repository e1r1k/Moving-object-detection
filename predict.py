"""
Run a trained U-Net on a REAL pair of consecutive frames.
 
    python predict.py --ckpt runs/unet1/best.pt \
        --frame-a "frigate/processed/Capture_00267_....png" \
        --frame-b "frigate/processed/Capture_00268_....png" --out results/f268
 
Pipeline (identical preprocessing to training):
  1. load both frames (grayscale), crop the 16 px border, integer-align B to A
  2. cut the aligned pair into overlapping 256x256 tiles, build the (A, B, B-A) input per tile
  3. run the network, merge tile probabilities with a smooth window (weighted average)
  4. threshold, label connected blobs per channel (object-in-A, object-in-B), drop tiny blobs
  5. pair A-blobs with B-blobs (a real mover appears in BOTH frames; a one-frame blob
     is more likely a cosmic ray / hot pixel / artifact)
 
Writes to --out: detections.json, overlay.png (aligned frame B, A blobs green / B blobs red),
crops.png (A | B | difference around the strongest detections).
Coordinates in the table / JSON are pixel positions in the ORIGINAL PNG of that frame
(x to the right, y downwards), so you can open the file and look there.
"""
from __future__ import annotations
 
import argparse
import json
import os
 
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
 
import synthetic_moving_objects as S
from train_unet import UNet, pick_device
 
 
# --------------------------------------------------------------------------
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)  
    a = ck["args"]
    model = UNet(base=a["base"], depth=a["depth"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
 
 
def tile_starts(length, patch, stride):
    starts = list(range(0, max(length - patch, 0) + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)        
    return starts
 
 
@torch.no_grad()
def predict_probability(model, A, B, device, patch=256, stride=128, batch=16, progress=True):
    """A, B: aligned uint8 arrays (H, W). Returns probability maps (2, H, W) float32."""
    H, W = A.shape
    if H < patch or W < patch:
        raise ValueError(f"frame {H}x{W} smaller than patch {patch}")
    h = np.hanning(patch + 2)[1:-1]
    win = (np.outer(h, h) + 1e-3).astype(np.float32)      
    acc = np.zeros((2, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    tiles = [(y, x) for y in tile_starts(H, patch, stride) for x in tile_starts(W, patch, stride)]
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = np.stack([S.to_input(A[y:y + patch, x0:x0 + patch], B[y:y + patch, x0:x0 + patch]) for y, x0 in chunk])
        prob = torch.sigmoid(model(torch.from_numpy(x).to(device)).float()).cpu().numpy()
        for (y, x0), p in zip(chunk, prob):
            acc[:, y:y + patch, x0:x0 + patch] += p * win
            wsum[y:y + patch, x0:x0 + patch] += win
        if progress:
            print(f"  tiles {min(i + batch, len(tiles))}/{len(tiles)}", end="\r", flush=True)
    if progress:
        print()
    return acc / wsum
 
 
def find_blobs(prob, thr, min_size):
    """Connected blobs above thr in each channel. Returns list of dicts (aligned-frame coordinates)."""
    blobs = []
    for c, name in ((0, "A"), (1, "B")):
        lab, n = ndi.label(prob[c] > thr, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        sizes = ndi.sum(np.ones_like(lab), lab, idx)
        cents = ndi.center_of_mass(np.ones_like(lab), lab, idx)
        maxp = ndi.maximum(prob[c], lab, idx)
        meanp = ndi.mean(prob[c], lab, idx)
        slices = ndi.find_objects(lab)                 
        for k in range(n):
            if sizes[k] >= min_size:
                sl = slices[k]
                blobs.append(dict(frame=name, x=float(cents[k][1]), y=float(cents[k][0]), size_px=int(sizes[k]),
                                  max_prob=float(maxp[k]), mean_prob=float(meanp[k]),
                                  bbox=[int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)]))
    return blobs
 
 
def pair_blobs(blobs, max_disp):
    """Greedy nearest-neighbour pairing of A blobs with B blobs (distance <= max_disp px)."""
    A = [b for b in blobs if b["frame"] == "A"]
    B = [b for b in blobs if b["frame"] == "B"]
    cand = sorted(((np.hypot(a["x"] - b["x"], a["y"] - b["y"]), i, j) for i, a in enumerate(A) for j, b in enumerate(B)))
    usedA, usedB, pairs = set(), set(), []
    for d, i, j in cand:
        if d > max_disp:
            break
        if i in usedA or j in usedB:
            continue
        usedA.add(i); usedB.add(j)
        pairs.append(dict(A=A[i], B=B[j], displacement_px=float(d),
                          dx=B[j]["x"] - A[i]["x"], dy=B[j]["y"] - A[i]["y"]))
    unpaired = [a for i, a in enumerate(A) if i not in usedA] + [b for j, b in enumerate(B) if j not in usedB]
    return pairs, unpaired
 
 
# --------------------------------------------------------------------------
def stretch(a):
    return np.clip(a.astype(np.float32) * 3, 0, 255).astype(np.uint8)
 
 
def save_overlay(path, B, blobs, r=14):
    im = Image.fromarray(stretch(B)).convert("RGB")
    d = ImageDraw.Draw(im)
    for b in blobs:
        col = (0, 255, 0) if b["frame"] == "A" else (255, 0, 0)
        rr = r + (4 if b["frame"] == "A" else 0)
        d.ellipse([b["x"] - rr, b["y"] - rr, b["x"] + rr, b["y"] + rr], outline=col, width=2)
    im.save(path)
 
 
def save_crops(path, A, B, items, half=32, scale=4):
    """items: list of (label, x, y) in aligned coordinates -> rows of A | B | 128+4*(B-A)."""
    rows = []
    H, W = A.shape
    for label, x, y in items:
        x0 = int(np.clip(round(x) - half, 0, W - 2 * half)); y0 = int(np.clip(round(y) - half, 0, H - 2 * half))
        a, b = A[y0:y0 + 2 * half, x0:x0 + 2 * half], B[y0:y0 + 2 * half, x0:x0 + 2 * half]
        d = np.clip(128 + 4 * (b.astype(np.float32) - a.astype(np.float32)), 0, 255).astype(np.uint8)
        row = np.hstack([stretch(a), stretch(b), d])
        row = np.kron(row, np.ones((scale, scale), np.uint8))
        im = Image.fromarray(row).convert("RGB")
        ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0))
        rows.append(np.asarray(im))
    if rows:
        Image.fromarray(np.vstack(rows)).save(path)
 
 
# --------------------------------------------------------------------------
def run(model, frame_a, frame_b, device, thr=0.5, min_size=2, patch=256, stride=128, batch=16,
        max_disp=120.0, align=True):
    pairs_src = S.FramePairs([frame_a, frame_b], gap=1, align=align, require_consecutive=False)
    # FramePairs sorts its paths; make sure A is really the earlier frame the caller meant
    ordered = pairs_src.paths
    if ordered[0] != frame_a:
        print("note: frames re-ordered by filename so that A is the earlier frame")
    A, B = pairs_src.get(0)
    dy, dx = pairs_src._shift.get(0, (0, 0))
    border = pairs_src.border
    offA = (max(0, dx) + border, max(0, dy) + border)              
    offB = (offA[0] - dx, offA[1] - dy)                        
    print(f"aligned crop {A.shape[1]}x{A.shape[0]}, integer shift of B (dy, dx) = ({dy}, {dx})")
    if max(abs(dy), abs(dx)) > 4:
        print("  warning: large shift - these frames are far apart in time; rotation is NOT corrected, "
              "so expect more star residuals. Consecutive frames work best.")
    prob = predict_probability(model, A, B, device, patch, stride, batch)
    blobs = find_blobs(prob, thr, min_size)
    pairs, unpaired = pair_blobs(blobs, max_disp)

    for b in blobs:
        ox, oy = offA if b["frame"] == "A" else offB
        b["x_png"], b["y_png"] = b["x"] + ox, b["y"] + oy
    return dict(A=A, B=B, prob=prob, blobs=blobs, pairs=pairs, unpaired=unpaired, offsets=dict(A=offA, B=offB),
                shift=(dy, dx))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame-a", required=True, help="earlier frame (PNG)")
    ap.add_argument("--frame-b", required=True, help="the next consecutive frame (PNG)")
    ap.add_argument("--out", default="predict_out")
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold (lower = more sensitive, more false blobs)")
    ap.add_argument("--min-size", type=int, default=2, help="minimum blob size in pixels")
    ap.add_argument("--patch", type=int, default=256, help="must match the training patch size")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=120.0, help="max A->B blob distance (px) to count as the same object")
    ap.add_argument("--top", type=int, default=8, help="number of detections to show in crops.png")
    ap.add_argument("--no-align", action="store_true")
    a = ap.parse_args()
 
    os.makedirs(a.out, exist_ok=True)
    device = pick_device()
    model, ck = load_model(a.ckpt, device)
    print(f"device {device}; checkpoint epoch {ck['epoch'] + 1}, val F1 {ck['best']:.3f}")
    r = run(model, a.frame_a, a.frame_b, device, a.thr, a.min_size, a.patch, a.stride, a.batch, a.max_disp, not a.no_align)
 
    pairs, unpaired = r["pairs"], r["unpaired"]
    pairs.sort(key=lambda p: -(p["A"]["max_prob"] + p["B"]["max_prob"]))
    print(f"\nRESULT: {len(pairs)} candidate moving object(s) seen in BOTH frames, {len(unpaired)} unpaired blob(s)")
    if pairs:
        print("  #   A (x,y)          B (x,y)        moved   size A/B   max prob A/B")
        for i, p in enumerate(pairs, 1):
            A_, B_ = p["A"], p["B"]
            print(f"  {i:<3d} ({A_['x_png']:7.1f},{A_['y_png']:7.1f})  ({B_['x_png']:7.1f},{B_['y_png']:7.1f})  "
                  f"{p['displacement_px']:5.1f}px  {A_['size_px']:3d}/{B_['size_px']:<3d}   {A_['max_prob']:.2f}/{B_['max_prob']:.2f}")
    if unpaired:
        print(f"  unpaired (one frame only - often cosmic rays or residuals): "
              + ", ".join(f"{b['frame']}({b['x_png']:.0f},{b['y_png']:.0f}) p={b['max_prob']:.2f}" for b in unpaired[:10])
              + (" ..." if len(unpaired) > 10 else ""))
 
    with open(os.path.join(a.out, "detections.json"), "w") as f:
        json.dump(dict(frame_a=a.frame_a, frame_b=a.frame_b, thr=a.thr, min_size=a.min_size,
                       integer_shift_dy_dx=r["shift"], aligned_crop_origin_in_png=r["offsets"],
                       pairs=pairs, unpaired=unpaired), f, indent=2)
    save_overlay(os.path.join(a.out, "overlay.png"), r["B"], r["blobs"])
    items = [(f"pair {i + 1}: A p={p['A']['max_prob']:.2f}  B p={p['B']['max_prob']:.2f}", p["B"]["x"], p["B"]["y"])
             for i, p in enumerate(pairs[:a.top])]
    items += [(f"unpaired {b['frame']} p={b['max_prob']:.2f}", b["x"], b["y"]) for b in unpaired[:max(0, a.top - len(items))]]
    save_crops(os.path.join(a.out, "crops.png"), r["A"], r["B"], items)
    print(f"\nsaved detections.json, overlay.png, crops.png to {a.out}")
 
 
if __name__ == "__main__":
    main()