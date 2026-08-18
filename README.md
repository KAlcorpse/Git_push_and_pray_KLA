# AI-Based Restoration of Degraded Images for Semiconductor Inspection

KLA problem statement, Hackathon 2026 (SEMICON India), Phase 1 submission.

Restores 128×128 degraded `NoisyLR` arrays to 256×256 in a single forward pass
of a 6.81 M-parameter MiniRestormer: a channel-attention transformer U-Net that
predicts a correction on top of a bicubic upsample.

**Held-out result: 28.15 dB PSNR / 0.7692 SSIM / 0.1439 LPIPS**, measured on 320
images excluded from both training and model selection.

---

## 1. Run the code

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

That is the whole contract. No source edits, No notebook cells, no local paths,
no configuration, no interaction. The output directory is created if it does not
exist. The default checkpoint at `models/best.pth` is resolved relative to
`run.py`, so the command works from any working directory.

Example:

```bash
python run.py /data/test/NoisyLR ./restored
```

### Input and output contract

| | |
|---|---|
| input | every `*.npy` in `<input-dir>`; shape `(H, W)`, `(H, W, 1)` or `(1, H, W)` |
| output | one `*.npy` per input in `<output-dir>`, **same filename** |
| output shape | `(2H, 2W)`, 2-D, grayscale |
| output dtype | `float32` |
| output values | clipped to `[0, 1]`, guaranteed free of NaN and Inf |
| mixed sizes | fine; inputs are grouped by shape so batching stays valid |

On completion `run.py` prints the observed value range, a count of non-finite
values (expected: 0), and a full end-to-end timing breakdown, so the run itself
is the evidence that the contract holds.

### Optional flags (none is required)

```
--weights PATH      alternate checkpoint             default models/best.pth
--batch-size N      0 = size automatically from free VRAM   default 0
--device cuda|cpu   default cuda when available
--tile N            process in N×N tiles with Hann blending; 0 = whole image
```

`--tile 128` keeps inference at the resolution the model was validated on for
very large inputs. It is not needed for 128×128 or 256×256 inputs, which run
unchanged: attention is over channels, not pixels, so no layer has a fixed
spatial size.

---

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Developed on Python 3.12, PyTorch 2.x, CUDA 12.x. **Inference imports only
`torch` and `numpy`** — the remaining entries in `requirements.txt` are for
training and evaluation and are never touched by `run.py`. Nothing is downloaded
at run time, so the pipeline works on a machine with no internet access once the
environment is installed.

---

## 3. Results

Held-out split: 320 images, `val_split=0.1`, `seed=42`, excluded from training
**and** from model selection.

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | HF ratio |
|---|---|---|---|---|
| baseline: bicubic upsample of the noisy input | 22.71 | 0.5285 | — | — |
| **submitted model** | **28.15** | **0.7692** | **0.1439** | **0.500** |
| reference: perfect denoising + bicubic upsample | 32.10 | 0.8727 | — | 1.000 |

The two reference rows are measured, not assumed. The lower row is what doing
nothing scores; the upper row is what a *flawless* denoiser scores, obtained by
downsampling the ground truth and bicubic-upsampling it back. The submitted
model closes **58% of the available PSNR range and 70% of the SSIM range**.

`HF ratio` is an additional metric we defined: high-frequency energy above
0.25 cycles/px relative to ground truth. It exists because both PSNR and SSIM
*reward* blur — blurring a baseline improves PSNR by 2.6 dB while cutting
texture to a fifth of ground truth — so a distortion-only score cannot tell a
good restoration from a smeared one. It was used alongside LPIPS to choose the
final model, and the selection criterion during training was `SSIM − LPIPS`.

Ablation of the loss design on the same split:

| model | PSNR | SSIM | LPIPS | HF ratio |
|---|---|---|---|---|
| distortion-only loss | 28.17 | 0.7745 | 0.2543 | 0.450 |
| + LPIPS at weight 0.10 | 28.08 | 0.7662 | 0.1458 | 0.477 |
| **+ per-level heads and wide head (submitted)** | **28.15** | **0.7692** | **0.1439** | **0.500** |

The distortion-only model has the best PSNR and SSIM and 77% worse LPIPS with
visibly more smearing. The submitted model gives up 0.02 dB to halve perceptual
distance, and beats the equivalent single-head run on all four metrics at
identical FLOPs. Thus it can be concluded that the distortion model, sacrifices on LPIPS to boost other IQA metrics.

Full metric tables, restored examples and the failure case are in `results/`.

---

## 4. Runtime

| | |
|---|---|
| measurement | 152 images/s |
| batch size | 32 |
| precision | fp32 |
| hardware | NVIDIA laptop GPU 3050, 3.67 GiB usable, CUDA 12.x |
| software | Python 3.12, PyTorch 2.x, NumPy 2.x |
| timing method | `time.perf_counter()` around the entire pipeline, with an explicit `torch.cuda.synchronize()` before the device-to-host copy is timed |

The timed window follows the brief's definition exactly: disk read,
preprocessing, CPU→GPU transfer, model execution, GPU→CPU transfer,
post-processing and saving. `run.py` prints this breakdown on every run, so the
number above is reproducible rather than asserted.

**Throughput decisions, both measured rather than assumed:**

- *No test-time augmentation.* An 8× dihedral self-ensemble was measured at
  +0.065 dB and +0.0024 SSIM for 8× the compute.
- *One model, not two.* Blending with a second architecture gains +0.36 dB but
  costs 2.2× LPIPS and halves throughput.

---

## 5. Reproducing the checkpoint

```bash
cd train
RUN_DIR=runs/dim64_arch DIM=64 NUM_BLOCKS=2,2,2 NUM_HEADS=1,2,4 WIDE_HEAD=32 \
  W_CHAR=0.5 W_SSIM=0.35 W_GRAD=0.15 W_LPIPS=0.1 \
  MAX_HOURS=2 BATCH=6 python train.py
```

Every configuration value is an environment-variable override, so no experiment
requires a source edit. Edit `GT_DIR` and `NOISY_DIR` at the top of `train.py`
to point at the dataset; nothing else needs changing.

| | |
|---|---|
| architecture | `dim=64`, `blocks=(2,2,2)`, heads `(1,2,4)`, wide head 32 — 6.81 M parameters |
| input | 64×64 random crops of NoisyLR paired with the exactly-corresponding 128×128 of GT |
| augmentation | dihedral group, 8 exact pixel permutations, applied identically to both halves of the pair |
| loss | `0.50·Charbonnier + 0.35·(1−SSIM) + 0.15·gradient + 0.10·LPIPS` |
| optimiser | AdamW, lr 1e-4, weight decay 1e-4, 3-epoch warmup then cosine to 1e-6 |
| precision | fp32 (fp16 diverged on this data) |
| stabilisers | weight EMA 0.999, gradient clipping 1.0, loss-spike rejection |
| split | 90/10, `seed=42`, held out from training and from model selection |
| selection | best `SSIM − LPIPS` |

Training writes `best.pth`, `last.pth` and a per-epoch `train_log.csv`, and
resumes automatically from `last.pth` — delete it to start a fresh run.

The architecture is recovered from the checkpoint's **weight shapes** at load
time rather than from saved metadata, so a checkpoint can never silently be
loaded into the wrong model.

---

## 6. Repository layout

```
run.py                    inference entry point: run.py <input-dir> <output-dir>
requirements.txt          dependencies with pinned versions
README.md                 this file
models/
  __init__.py
  minirestormer.py        the architecture, imports only torch
  best.pth                submitted checkpoint
train/
  train.py                reproduces the submitted checkpoint
  dataset.py              paired loader, RAM cache, crops, dihedral augmentation
  degrade.py              the fitted forward degradation model
results/
  metrics.md              full metric tables and the ablation
  val_example.png         held-out image: input, bicubic, restored, ground truth
  test_example.png        test input: input, bicubic, restored
  failure_case.png        a below-average case, discussed in metrics.md
Solution_ppt.pdf
```

---

## 7. Assumptions and how the data is handled

- **NoisyLR values outside `[0, 1]` are handled intentionally.** They are signal,
  not error: they carry the speckle tail, measured across the corpus at −0.03 to
  +1.36. The input is passed to the model **unmodified** — no normalisation, no
  centring, no rescaling, no clipping. Only the **output** is clipped, inside our
  own pipeline, as the brief requires.
- Inputs are single-channel float arrays; outputs preserve the input filename.
- The output is exactly 2× the input in each spatial dimension.
- Only the three official degradation mechanisms are treated as benchmark
  requirements. We fitted them empirically rather than assuming an order:
  `y = downsample₂(x) · (1 + N(0, s²)) + N(0, g²)` with `s ≈ 0.163`,
  `g ≈ 0.004`, fitting the variance law at R² = 0.99 over 300 pairs. Speckle is
  Gaussian-multiplicative (kurtosis 3.03; a uniform speckle is rejected at
  p = 1e-20), and its level varies about 1.8× across images.
- No confidential, unlicensed or inaccessible data is used.

---

## 8. External resources

| resource | how it is used | licence |
|---|---|---|
| [`lpips`](https://github.com/richzhang/PerceptualSimilarity) (Zhang et al., CVPR 2018), AlexNet backbone | training loss and reported metric. **Not part of the inference pipeline** and not required to run `run.py` | BSD-2-Clause |
| [PyTorch](https://github.com/pytorch/pytorch) | framework | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | array I/O | BSD-3-Clause |
| [scikit-image](https://github.com/scikit-image/scikit-image) | SSIM during evaluation | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy) | degradation fitting | BSD-3-Clause |

No external datasets and no pretrained restoration weights were used. The model
is trained from scratch on the provided pairs only.

Additional implementation details

1.The complete restoration model is implemented using standard PyTorch modules; no external pretrained restoration architecture or weights are imported.
2.All trainable model parameters are initialized and optimized from scratch using only the provided training pairs.
3.External libraries are used only for numerical operations, evaluation metrics, degradation fitting, and supporting the training pipeline.
4.The inference pipeline is self-contained and does not require lpips or any pretrained model weights.
5.No external datasets, synthetic datasets from third-party sources, or additional restoration checkpoints were used.
6.Evaluation metrics are computed independently on the provided test/validation pairs to ensure a consistent comparison across methods.
