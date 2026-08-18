"""Restoration inference entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes one .npy per input to
<output-dir> under the same filename. Needs only torch and numpy: no internet,
no API keys, no extra downloads, no user interaction, no configuration to edit.

Optional flags, all with working defaults:
    --weights PATH     checkpoint            (default models/best.pth)
    --batch-size N     0 = size from free VRAM
    --device cuda|cpu  (default cuda when available)
    --tile N           process in NxN tiles; 0 = whole image
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from models import MiniRestormer, config_from_state_dict, load_state_dict_compat

SCALE = 2


# --------------------------------------------------------------------------
def load_model(weights, device):
    """Rebuild the exact trained architecture from the checkpoint itself.

    The config is recovered from the weight shapes rather than from saved
    metadata, so a checkpoint that predates a config key still loads correctly
    instead of silently building the wrong model.
    """
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = config_from_state_dict(sd, ck if isinstance(ck, dict) else None, SCALE)
    model = MiniRestormer(**cfg)
    load_state_dict_compat(model, sd)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model  : {cfg}", flush=True)
    print(f"weights: {weights}  ({n:.2f} M parameters)", flush=True)
    return model


def read_npy(path):
    """Load one input and normalise it to a 2-D float32 array.

    Accepts (H, W), (H, W, 1) and (1, H, W); anything else is a hard error
    rather than a silent guess.
    """
    a = np.load(path, allow_pickle=False)
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"{os.path.basename(path)}: expected a 2-D array, got shape {a.shape}")
    return a.astype(np.float32, copy=False)


@torch.no_grad()
def forward(model, x, tile=0):
    """x: (B,1,H,W) -> (B,1,2H,2W). Tiling keeps inference at the resolution the
    model was validated on for larger inputs, blending seams with a Hann ramp."""
    H, W = x.shape[-2:]
    if not tile or (H <= tile and W <= tile):
        return model(x).float()

    ov = max(8, tile // 8)
    stride, e = tile - ov, ov * SCALE
    out = torch.zeros(x.shape[0], 1, H * SCALE, W * SCALE, device=x.device)
    wsum = torch.zeros_like(out)
    ramp = torch.hann_window(2 * e, periodic=False, device=x.device)
    for y0 in range(0, max(H - tile, 0) + 1, stride):
        for x0 in range(0, max(W - tile, 0) + 1, stride):
            y0, x0 = min(y0, H - tile), min(x0, W - tile)
            patch = model(x[..., y0:y0 + tile, x0:x0 + tile]).float()
            w = torch.ones_like(patch)
            if y0 > 0:
                w[..., :e, :] *= ramp[:e].view(1, 1, -1, 1)
            if x0 > 0:
                w[..., :, :e] *= ramp[:e].view(1, 1, 1, -1)
            if y0 + tile < H:
                w[..., -e:, :] *= ramp[-e:].view(1, 1, -1, 1)
            if x0 + tile < W:
                w[..., :, -e:] *= ramp[-e:].view(1, 1, 1, -1)
            ys, xs = y0 * SCALE, x0 * SCALE
            out[..., ys:ys + tile * SCALE, xs:xs + tile * SCALE] += patch * w
            wsum[..., ys:ys + tile * SCALE, xs:xs + tile * SCALE] += w
    return out / wsum.clamp(min=1e-8)


def auto_batch(device, hw):
    if device.type != "cuda":
        return 4
    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    per = 0.35 * (hw / (128 * 128))          # GiB per image at inference, with headroom
    return int(max(1, min(64, (free * 0.6) / max(per, 1e-6))))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Restore degraded NoisyLR images.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best.pth"))
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t_start = time.perf_counter()                    # end-to-end clock starts here

    if not os.path.isdir(a.input_dir):
        sys.exit(f"input directory not found: {a.input_dir}")
    if not os.path.isfile(a.weights):
        sys.exit(f"weights not found: {a.weights}")
    os.makedirs(a.output_dir, exist_ok=True)

    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = load_model(a.weights, device)

    names = sorted(f for f in os.listdir(a.input_dir) if f.endswith(".npy"))
    if not names:
        sys.exit(f"no .npy files in {a.input_dir}")

    groups, t_read = {}, 0.0
    for f in names:
        t = time.perf_counter()
        arr = read_npy(os.path.join(a.input_dir, f))
        t_read += time.perf_counter() - t
        groups.setdefault(arr.shape, []).append((f, arr))

    print(f"input  : {len(names)} files, shapes "
          + ", ".join(f"{s[0]}x{s[1]} ({len(v)})" for s, v in groups.items()), flush=True)

    t_gpu = t_write = 0.0
    done, lo, hi, n_bad = 0, np.inf, -np.inf, 0

    for shape, items in groups.items():
        bs = a.batch_size or auto_batch(device, shape[0] * shape[1])
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]

            t = time.perf_counter()
            x = torch.from_numpy(np.stack([arr for _, arr in chunk])).unsqueeze(1)
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = forward(model, x, a.tile)
            # Guarantee the output contract on-device before it leaves the GPU.
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            y = y.cpu().numpy()
            t_gpu += time.perf_counter() - t

            t = time.perf_counter()
            for j, (f, _) in enumerate(chunk):
                out = np.ascontiguousarray(y[j, 0], dtype=np.float32)
                n_bad += int(np.count_nonzero(~np.isfinite(out)))
                lo, hi = min(lo, float(out.min())), max(hi, float(out.max()))
                np.save(os.path.join(a.output_dir, f), out)
            t_write += time.perf_counter() - t

            done += len(chunk)
            print(f"\r  restored {done}/{len(names)}", end="", flush=True)

    total = time.perf_counter() - t_start
    bs_used = a.batch_size or "auto"

    print(f"\n\nwrote {done} files to {a.output_dir}")
    print("--- output contract ---")
    print(f"  dtype float32, 2-D arrays, {SCALE}x the input in each dimension")
    print(f"  value range [{lo:.4f}, {hi:.4f}]      non-finite values: {n_bad}")
    print("--- end-to-end runtime ---")
    print(f"  disk read           {t_read:8.2f} s")
    print(f"  transfer + model    {t_gpu:8.2f} s")
    print(f"  post + save         {t_write:8.2f} s")
    print(f"  init + model load   {total - t_read - t_gpu - t_write:8.2f} s")
    print(f"  TOTAL               {total:8.2f} s   ->  {done / total:.1f} images/s")
    print(f"  device={device}  batch={bs_used}  tile={a.tile or 'off'}")
    if device.type == "cuda":
        print(f"  gpu={torch.cuda.get_device_name(0)}  "
              f"peak={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    print(f"  torch={torch.__version__}  numpy={np.__version__}  python={sys.version.split()[0]}")
    print("  timing method: time.perf_counter() around the whole pipeline, with an")
    print("  explicit CUDA synchronise before the device-to-host copy is timed.")


if __name__ == "__main__":
    main()
