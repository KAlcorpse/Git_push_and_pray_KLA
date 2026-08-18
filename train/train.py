import argparse
import os

# Must be set BEFORE torch initialises CUDA. Fixes the "epoch 1 fine, epoch 2
# OOM" fragmentation pattern: training crops and full-res validation allocate
# very differently-shaped blocks, and the default allocator can't reuse them.
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RestorationDataset
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MiniRestormer

# ---------------------------- configuration ----------------------------------
#     python train.py <gt-dir> <noisylr-dir>


_e = os.environ.get


def _cli():
    ap = argparse.ArgumentParser(
        description="Train the restoration model on paired GT / NoisyLR .npy directories.")
    ap.add_argument("gt_dir", nargs="?", default=_e("GT_DIR"),
                    help="directory of clean ground-truth .npy files (256x256)")
    ap.add_argument("noisy_dir", nargs="?", default=_e("NOISY_DIR"),
                    help="directory of degraded NoisyLR .npy files (128x128)")
    ap.add_argument("--out-dir", default=_e("RUN_DIR", "runs/dim64_arch"),
                    help="where best.pth, last.pth and train_log.csv are written")
    # parse_known_args, not parse_args: importing this module must never fail,
    # and unknown flags are left for anything that wraps it.
    return ap.parse_known_args()[0], ap


_ARGS, _AP = _cli()
GT_DIR = os.path.abspath(os.path.expanduser(_ARGS.gt_dir)) if _ARGS.gt_dir else None
NOISY_DIR = os.path.abspath(os.path.expanduser(_ARGS.noisy_dir)) if _ARGS.noisy_dir else None
OUT_DIR = _ARGS.out_dir
DIM = int(_e("DIM", 64))
_h = _e("NUM_HEADS", "1,2,4")                 
NUM_HEADS = tuple(int(x) for x in _h.split(",")) if "," in _h else int(_h)
NUM_BLOCKS = tuple(int(x) for x in _e("NUM_BLOCKS", "2,2,2").split(","))
LOG_INPUT = _e("LOG_INPUT", "0") == "1"
NOISE_COND = _e("NOISE_COND", "0") == "1"
WIDE_HEAD = int(_e("WIDE_HEAD", 32))          

EPOCHS = int(_e("EPOCHS", 400))
MAX_HOURS = float(_e("MAX_HOURS", 2.0))
CROP = int(_e("CROP", 64))
BATCH = int(_e("BATCH", 6))                   
FULLRES_EPOCHS = int(_e("FULLRES_EPOCHS", 4))
FULLRES_BATCH = int(_e("FULLRES_BATCH", 0))
VAL_BATCH = int(_e("VAL_BATCH", 2))

LR, MIN_LR = float(_e("LR", 1e-4)), 1e-6
WEIGHT_DECAY, WARMUP = 1e-4, int(_e("WARMUP", 3))
SYNTH_PROB, SYNTH_JITTER = float(_e("SYNTH_PROB", 0.0)), 0.2
VAL_SPLIT, SEED, WORKERS = 0.1, 42, int(_e("WORKERS", 2))
EMA_DECAY, CLIP_GRAD = 0.999, 1.0

# Loss weights. SSIM rewards blur, so it is capped; gradient and LPIPS push back.
W_CHAR  = float(_e("W_CHAR",  0.50))          # <-- was deleted, restore it
W_SSIM  = float(_e("W_SSIM",  0.35))
W_GRAD  = float(_e("W_GRAD",  0.15))
W_LPIPS = float(_e("W_LPIPS", 0.10))        
# ------------------------------------------------------------------------------

BATCH_CANDIDATES = [16, 12, 8, 6, 4, 3, 2, 1]


def probe_batch(dim, blocks, crop, safety_step=1):
    """Largest batch that survives two real optimizer steps, minus a safety step.

    Two steps, because AdamW allocates its moment buffers on the first .step().
    The step-down is deliberate: the probe runs on an empty GPU, whereas real
    training also holds the dataset cache and a fragmented heap.
    """
    if not torch.cuda.is_available():
        return 4
    for i, bs in enumerate(BATCH_CANDIDATES):
        model = opt = x = y = loss = None
        try:
            model = MiniRestormer(dim=dim, num_blocks=blocks, num_heads=NUM_HEADS,
                                  log_input=LOG_INPUT, noise_cond=NOISE_COND,
                                  wide_head=WIDE_HEAD).cuda()
            model = model.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(bs, 1, crop, crop, device="cuda").to(memory_format=torch.channels_last)
            y = torch.randn(bs, 1, crop * 2, crop * 2, device="cuda")
            for _ in range(2):
                loss = F.l1_loss(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            chosen = BATCH_CANDIDATES[min(i + safety_step, len(BATCH_CANDIDATES) - 1)]
            print(f"  crop {crop}: max batch {bs} (peak {peak:.2f} GiB) -> using {chosen}",
                  flush=True)
            return chosen
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" not in str(err).lower():
                raise
        finally:
            del model, opt, x, y, loss
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return 0


# ---------------- losses ----------------
def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


class SSIM(nn.Module):
    def __init__(self, window=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(window, dtype=torch.float32) - window // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        self.register_buffer("w", (g @ g.t()).view(1, 1, window, window))
        self.pad = window // 2

    def forward(self, a, b):
        w = self.w.to(a.dtype)
        mu1, mu2 = F.conv2d(a, w, padding=self.pad), F.conv2d(b, w, padding=self.pad)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = F.conv2d(a * a, w, padding=self.pad) - mu1_sq
        s2 = F.conv2d(b * b, w, padding=self.pad) - mu2_sq
        s12 = F.conv2d(a * b, w, padding=self.pad) - mu12
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mu12 + C1) * (2 * s12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))).mean()


SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def gradient_loss(pred, target):
    kx, ky = SOBEL_X.to(pred.device), SOBEL_Y.to(pred.device)
    return (F.l1_loss(F.conv2d(pred, kx, padding=1), F.conv2d(target, kx, padding=1)) +
            F.l1_loss(F.conv2d(pred, ky, padding=1), F.conv2d(target, ky, padding=1)))


_LPIPS, _LPIPS_OK = None, None


def lpips_loss(pred, target):
    """Perceptual distance. Auto-disables (returns 0) if the package or its
    pretrained weights are unavailable -- never kills an overnight run."""
    global _LPIPS, _LPIPS_OK
    if _LPIPS_OK is False:
        return torch.zeros((), device=pred.device)
    if _LPIPS is None:
        try:
            import lpips as _lp
            _LPIPS = _lp.LPIPS(net="alex", verbose=False).to(pred.device).eval()
            for p in _LPIPS.parameters():
                p.requires_grad_(False)
            _LPIPS_OK = True
            print("  LPIPS loss active (alex)", flush=True)
        except Exception as err:                                   # noqa: BLE001
            print(f"  LPIPS unavailable ({type(err).__name__}); continuing without it",
                  flush=True)
            _LPIPS_OK = False
            return torch.zeros((), device=pred.device)
    # grayscale -> 3 channels, [0,1] -> [-1,1], which is what LPIPS expects
    return _LPIPS(pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1,
                  target.repeat(1, 3, 1, 1) * 2 - 1).mean()


def hf_energy(t, cut=0.25):
    """Energy above `cut` cycles/pixel -- a direct proxy for fine texture."""
    f = torch.fft.rfft2(t.float(), norm="ortho").abs()
    H, W = t.shape[-2:]
    fy = torch.fft.fftfreq(H, device=t.device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(W, device=t.device).view(1, -1)
    return float((f * ((fy ** 2 + fx ** 2).sqrt() > cut)).sum())


# ---------------- EMA ----------------
class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        if not all(torch.isfinite(v).all() for v in sd.values() if v.dtype.is_floating_point):
            return                      # never average in a corrupted step
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.other[k] = v.detach().clone()

    def state_dict(self):
        return {**{k: v.clone() for k, v in self.shadow.items()},
                **{k: v.clone() for k, v in self.other.items()}}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            target = self.shadow if k in self.shadow else self.other
            target[k] = v.detach().clone().float() if k in self.shadow else v.detach().clone()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def validate(model, ds, device, batch):
    """Returns PSNR, SSIM, LPIPS, and the HF ratio (texture kept vs ground truth).
    Plain loop, no DataLoader workers -- one less thing to leak or OOM."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    p = s = lp = n = 0.0
    hf_p = hf_g = 0.0
    for start in range(0, len(ds), batch):
        items = [ds[i] for i in range(start, min(start + batch, len(ds)))]
        noisy = torch.stack([a for a, _ in items]).to(device)
        gt_t = torch.stack([b for _, b in items]).to(device)
        pred_t = model(noisy).float().clamp(0, 1)
        if W_LPIPS > 0:
            lp += float(lpips_loss(pred_t, gt_t)) * pred_t.shape[0]
        hf_p += hf_energy(pred_t.cpu())
        hf_g += hf_energy(gt_t.cpu())
        pred, gt = pred_t.cpu().numpy(), gt_t.cpu().numpy()
        for i in range(pred.shape[0]):
            p += psnr(gt[i, 0], pred[i, 0])
            s += structural_similarity(gt[i, 0], pred[i, 0], data_range=1.0)
            n += 1
        del noisy, gt_t, pred_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p / n, s / n, lp / n, hf_p / max(hf_g, 1e-9)


def main():
    if not GT_DIR or not NOISY_DIR:
        _AP.error("give both data directories: python train.py <gt-dir> <noisylr-dir>")
    for d, what in ((GT_DIR, "GT"), (NOISY_DIR, "NoisyLR")):
        if not os.path.isdir(d):
            _AP.error(f"{what} directory not found: {d}")
    print(f"data   : GT {GT_DIR}\n         NoisyLR {NOISY_DIR}", flush=True)
    global BATCH, FULLRES_BATCH
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device}  out={OUT_DIR}  dim={DIM} blocks={NUM_BLOCKS}", flush=True)
    print(f"loss: char={W_CHAR} ssim={W_SSIM} grad={W_GRAD} lpips={W_LPIPS}", flush=True)

    if BATCH == 0:
        BATCH = probe_batch(DIM, NUM_BLOCKS, CROP)
        if BATCH == 0:
            raise SystemExit("does not fit even at batch 1 -- lower DIM or NUM_BLOCKS")
    if FULLRES_BATCH == 0 and FULLRES_EPOCHS > 0:
        FULLRES_BATCH = probe_batch(DIM, NUM_BLOCKS, CROP * 2)
    print(f"batch={BATCH}  fullres_batch={FULLRES_BATCH}", flush=True)

    train_ds = RestorationDataset(GT_DIR, NOISY_DIR, "train", VAL_SPLIT, SEED,
                                  crop=CROP, synth_prob=SYNTH_PROB,
                                  synth_jitter=SYNTH_JITTER)
    val_ds = RestorationDataset(GT_DIR, NOISY_DIR, "val", VAL_SPLIT, SEED,
                                crop=None, augment=False)

    ARCH = dict(dim=DIM, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS,
                log_input=LOG_INPUT, noise_cond=NOISE_COND, wide_head=WIDE_HEAD)
    model = MiniRestormer(**ARCH).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.3f} M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = ModelEMA(model, EMA_DECAY)
    ssim_fn = SSIM().to(device)

    start_epoch, best_score, best_psnr, best_ssim = 1, -1.0, 0.0, 0.0
    resume = os.path.join(OUT_DIR, "last.pth")
    if os.path.isfile(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        start_epoch = ck["epoch"] + 1
        best_score = ck.get("best_score", -1.0)
        best_psnr, best_ssim = ck.get("best_psnr", 0.0), ck.get("best_ssim", 0.0)
        print(f"resumed at epoch {start_epoch}; delete {resume} to start fresh", flush=True)

    log = os.path.join(OUT_DIR, "train_log.csv")
    if not os.path.exists(log):
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "loss", "psnr", "ssim",
                                    "ema_psnr", "ema_ssim", "minutes",
                                    "lpips", "hf_ratio"])

    def make_loader(crop, bs):
        train_ds.set_crop(crop)
        return DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=WORKERS,
                          pin_memory=device.type == "cuda", drop_last=True,
                          persistent_workers=WORKERS > 0)

    crop_loader = make_loader(CROP, BATCH)
    fullres_loader = None
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        fullres = FULLRES_EPOCHS > 0 and epoch > EPOCHS - FULLRES_EPOCHS
        if fullres:
            if fullres_loader is None:
                del crop_loader
                gc.collect()
                fullres_loader = make_loader(None, max(FULLRES_BATCH, 1))
            train_loader, bs = fullres_loader, max(FULLRES_BATCH, 1)
        else:
            train_ds.set_crop(CROP)
            train_loader, bs = crop_loader, BATCH

        if epoch <= WARMUP:
            lr = LR * epoch / WARMUP
        else:
            by_epoch = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
            by_time = (time.time() - t0) / (MAX_HOURS * 3600)
            prog = min(1.0, max(by_epoch, by_time))
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        total = count = skipped = 0
        recent = []
        for noisy, gt in tqdm(train_loader, desc=f"epoch {epoch} bs={bs}", mininterval=10):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            if device.type == "cuda":
                noisy = noisy.to(memory_format=torch.channels_last)
                gt = gt.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            out = model(noisy).float()
            gt32 = gt.float()
            loss = (W_CHAR * charbonnier(out, gt32) +
                    W_SSIM * (1 - ssim_fn(out.clamp(0, 1), gt32)) +
                    W_GRAD * gradient_loss(out, gt32))
            if W_LPIPS > 0:
                loss = loss + W_LPIPS * lpips_loss(out, gt32)

            l = loss.item()
            med = sorted(recent[-50:])[len(recent[-50:]) // 2] if len(recent) >= 20 else None
            if not math.isfinite(l) or (med is not None and l > 3 * med):
                skipped += 1                     # one bad batch must not nuke the run
                continue
            recent.append(l)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            ema.update(model)
            total += l * noisy.size(0)
            count += noisy.size(0)

        train_loss = total / max(count, 1)
        val_p, val_s, val_lp, val_hf = validate(model, val_ds, device, VAL_BATCH)
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        ema_p, ema_s, ema_lp, ema_hf = validate(model, val_ds, device, VAL_BATCH)
        model.load_state_dict(raw)

        mins = (time.time() - t0) / 60
        print(f"epoch {epoch}: loss={train_loss:.5f}  "
              f"val {val_p:.2f}dB/{val_s:.4f} lpips {val_lp:.4f} hf {val_hf:.3f}  |  "
              f"ema {ema_p:.2f}dB/{ema_s:.4f} lpips {ema_lp:.4f} hf {ema_hf:.3f}  "
              f"({mins:.1f} min)" + (f"  [{skipped} spikes]" if skipped else ""), flush=True)

        use_ema = ema_s >= val_s
        cp, cs, clp, chf = (ema_p, ema_s, ema_lp, ema_hf) if use_ema else (val_p, val_s, val_lp, val_hf)
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.2e}", f"{train_loss:.6f}",
                                    f"{val_p:.4f}", f"{val_s:.5f}",
                                    f"{ema_p:.4f}", f"{ema_s:.5f}", f"{mins:.1f}",
                                    f"{clp:.5f}", f"{chf:.4f}"])

        # Selection score: SSIM, minus LPIPS if we are optimising perceptually.
        # Pure-SSIM selection would pick the blurriest checkpoint.
        score = cs - (clp if W_LPIPS > 0 else 0.0)
        if score > best_score:
            best_score, best_psnr, best_ssim = score, cp, cs
            torch.save({"model": ema.state_dict() if use_ema else raw,
                        "is_ema": bool(use_ema), "epoch": int(epoch),
                        "val_psnr": float(cp), "val_ssim": float(cs),
                        "val_lpips": float(clp), "val_hf_ratio": float(chf),
                        "dim": DIM, "num_heads": NUM_HEADS,
                        "num_blocks": list(NUM_BLOCKS),
                        "log_input": LOG_INPUT, "noise_cond": NOISE_COND,
                        "wide_head": WIDE_HEAD},
                       os.path.join(OUT_DIR, "best.pth"))
            print(f"  -> best {'ema' if use_ema else 'raw'}  "
                  f"{cp:.2f} dB / {cs:.4f} / lpips {clp:.4f} / hf {chf:.3f}", flush=True)

        torch.save({"model": raw, "opt": opt.state_dict(), "ema": ema.state_dict(),
                    "epoch": int(epoch), "best_score": float(best_score),
                    "best_psnr": float(best_psnr), "best_ssim": float(best_ssim)},
                   os.path.join(OUT_DIR, "last.pth"))

        elapsed = time.time() - t0
        if elapsed + elapsed / (epoch - start_epoch + 1) > MAX_HOURS * 3600:
            print(f"stopping: next epoch would exceed the {MAX_HOURS}h budget", flush=True)
            break

    print(f"done. best {best_psnr:.2f} dB / {best_ssim:.4f} "
          f"({(time.time()-t0)/3600:.2f} h)  ->  {OUT_DIR}/best.pth", flush=True)


if __name__ == "__main__":
    main()
