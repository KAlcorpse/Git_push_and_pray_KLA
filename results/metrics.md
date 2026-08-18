# Results

All numbers are measured on the same held-out split: 320 images, `val_split=0.1`,
`seed=42`, excluded from training **and** from model selection. Every model in
these tables was scored by one code path so the comparison is like for like.

---

## 1. Headline result against a measured range

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | HF ratio |
|---|---|---|---|---|
| baseline: bicubic upsample of the noisy input | 22.71 | 0.5285 | — | — |
| **submitted model** | **28.15** | **0.7692** | **0.1439** | **0.500** |
| reference: perfect denoising + bicubic upsample | 32.10 | 0.8727 | — | 1.000 |

Both reference rows are measured rather than assumed. The lower row is what
doing nothing scores. The upper row is what a *flawless* denoiser scores: the
ground truth downsampled and bicubic-upsampled back, so it isolates exactly how
much of the task is the 2× upsample and how much is denoising.

The gap between them is 9.39 dB, and the submitted model closes **58% of the
PSNR range and 70% of the SSIM range**. The upsample itself costs very little,
which is why effort went into the noise model and the loss rather than into the
upsampling path.

---

## 2. Ablation

| run | change | PSNR | SSIM | LPIPS | HF ratio |
|---|---|---|---|---|---|
| inherited starting point | — | 27.08 | 0.7481 | — | — |
| honest split + fixed output-layer init | first valid measurement | 27.54 | 0.7524 | — | — |
| dim 48 | width | 27.93 | 0.7672 | — | — |
| dim 64 | width | 28.17 | 0.7745 | 0.2543 | 0.450 |
| dim 64, blocks (4,4,4) | depth | 28.17 | 0.7751 | — | — |
| + LPIPS at weight 0.10 | perceptual term | 28.08 | 0.7662 | 0.1458 | 0.477 |
| + LPIPS at weight 0.40 | perceptual term | 27.72 | 0.7516 | 0.1321 | 0.509 |
| **+ per-level heads (1,2,4) and wide head 32** | **submitted** | **28.15** | **0.7692** | **0.1439** | **0.500** |

Notes on the two rows that look like non-results, because they are the useful
ones:

- **Depth did nothing.** `(4,4,4)` matched `(2,2,2)` to two decimal places while
  taking 80% more wall-clock. Width was the lever; depth was not.
- **LPIPS saturates.** Quadrupling its weight from 0.10 to 0.40 bought 0.032
  LPIPS and cost 0.36 dB and 0.015 SSIM. The first 0.10 of weight captured
  almost all of the available perceptual gain.

The submitted model was chosen because it wins on all four metrics against the
equivalent single-head run at identical parameter count and FLOPs, and gives up
0.02 dB against the distortion-only model to halve perceptual distance. With
the scoring weights undisclosed, the model that is strong on all three reported
metrics is the defensible choice.

---

## 3. The second architecture: TinyNAFNet, and the ensemble

We did not settle on a single architecture by assumption. A second model was
built and trained independently by the other half of the team, and the two were
then combined and measured before a decision was taken.

### 3.1 What was built

**TinyNAFNet**, 3.78 M parameters: `base_ch=64`, 2 NAFBlocks per encoder and
decoder stage, 6 at the bottleneck, `expand_ratio=2`, `upscale_factor=2`. It
follows the NAFNet family (Chen et al., ECCV 2022, *Simple Baselines for Image
Restoration*): no explicit activation function, a **SimpleGate** that splits the
channels in half and multiplies them, and **simplified channel attention** in
place of a full squeeze-and-excitation block. U-Net skips are concatenated and
projected, and the model predicts the output directly rather than a residual on
a bicubic upsample.

It is a genuinely different design from MiniRestormer, not a variant: convolutional
rather than attention-based, gated rather than activated, concatenating rather
than adding, and roughly half the parameter count. That independence is what
makes the comparison and the ensemble worth measuring.

Source: `github.com/sanjayganeshprabu/KLA_sanjay` (`tinynafnet_model.py`,
`hybrid_ensemble.py`, `hybrid_pipeline.ipynb`).

### 3.2 The two models alone

Scored over 320 validation pairs by the ensemble script's `--tune` mode:

| model | params | PSNR ↑ | SSIM ↑ | LPIPS ↓ * |
|---|---|---|---|---|
| TinyNAFNet | 3.78 M | **28.337** | 0.7615 | 0.363 |
| MiniRestormer (submitted) | 6.81 M | 28.080 | **0.7690** | **0.131** |

\* LPIPS was instrumented on 3 of those 320 images, so treat its magnitude as an
estimate. Its ordering is not in doubt: it is monotone in the blend weight on
every one of the three, and the two models are 2.8× apart.

TinyNAFNet is **0.26 dB ahead on PSNR** and behind on both SSIM and LPIPS. That
is the distortion-versus-perception split of section 4 appearing between two
architectures rather than between two loss weightings: the model with the better
pixel error is the visibly smoother one.

**Split caveat.** These numbers come from an independently drawn 320-image
split (`random.sample` after `random.seed(42)`) rather than ours (the tail of a
seeded shuffle). We checked the overlap: **302 of 320 images are common, 94.4%**.
The 18 that differ were in our training set, which would flatter MiniRestormer
slightly, so the residual bias works *against* the ensemble rather than for it.
Absolute values on this split therefore differ a little from section 1, but every
row within the sweep shares one split and is directly comparable.

### 3.3 The blend

Outputs were combined as `w · MiniRestormer + (1 − w) · TinyNAFNet` and swept:

| w on MiniRestormer | PSNR ↑ | SSIM ↑ | LPIPS ↓ * |
|---|---|---|---|
| 0.0 (TinyNAFNet alone) | 28.337 | 0.7615 | 0.363 |
| 0.1 | 28.390 | 0.7648 | 0.341 |
| 0.2 | 28.425 | 0.7676 | — |
| **0.3 (best PSNR)** | **28.443** | 0.7697 | 0.283 |
| 0.4 | 28.442 | 0.7712 | — |
| 0.5 | 28.424 | 0.7721 | 0.223 |
| **0.6 (best SSIM)** | 28.388 | **0.7725** | — |
| 0.7 | 28.335 | 0.7723 | 0.175 |
| 0.8 | 28.265 | 0.7717 | — |
| 0.9 | 28.180 | 0.7705 | 0.142 |
| **1.0 (MiniRestormer alone, submitted)** | 28.080 | 0.7690 | **0.131** |

The blend beats **both** solo models on PSNR, peaking at +0.36 dB over
MiniRestormer alone. That gain is real: averaging two independent predictors
reduces error variance, which is exactly what mean-squared error rewards.

### 3.4 Why we ship one model

The same averaging that reduces error variance also averages away the
high-frequency detail the two models disagree on. Expressed as what each step
toward the blend actually costs:

| w | ΔPSNR | ΔSSIM | ΔLPIPS | dB gained per unit LPIPS lost |
|---|---|---|---|---|
| 1.0 | — | — | — | — |
| 0.9 | +0.100 | +0.0015 | +8% | 9.3 |
| 0.8 | +0.185 | +0.0027 | +21% | 6.8 |
| 0.7 | +0.255 | +0.0033 | +33% | 5.8 |
| 0.5 | +0.344 | +0.0031 | +71% | 3.7 |
| 0.3 | +0.363 | +0.0007 | +116% | 2.4 |

Three things decided it:

1. **The trade rate is monotone.** Every step toward `w = 1.0` is a better deal
   than the one before, so there is no interior optimum to defend. `w = 0.7` is
   a far better blend point than `w = 0.3`, but the same reasoning that prefers
   0.7 over 0.3 prefers 0.9 over 0.7, and 1.0 over 0.9.
2. **The perceptual cost is large where the distortion gain is small.** At the
   PSNR-optimal `w = 0.3` the gain is +1.3% PSNR and +0.09% SSIM against a
   **doubling** of perceptual distance. With KLA's metric weights undisclosed,
   that only pays off if LPIPS carries almost no weight.
3. **Throughput is a scored axis and the cost is flat in `w`.** Two forward
   passes per image roughly halves throughput at `w = 0.9` just as much as at
   `w = 0.3`, for a tenth of the quality gain.

The ensemble is reported as a measured negative result rather than omitted. It
was built, swept over 320 images, and not shipped.

### 3.5 One hypothesis, flagged as unproven

The two models were trained with different augmentation policies. TinyNAFNet's
used continuous rotation (±8°), zoom (0.9–1.15) and intensity scaling
(×0.9–1.2). We measured what those transforms do to the speckle by applying them
to a real pair and recomputing the residual `NoisyLR − ↓(GT)`:

| transform | noise σ | lag-1 autocorrelation |
|---|---|---|
| as delivered | 1.00× | −0.03 (white) |
| flips | 1.00× | −0.03 |
| intensity ×1.2 | 1.20× (signal scales too) | −0.03 |
| rotation 8° | 0.65× | +0.24 |
| zoom 1.15 | 0.89× | +0.53 |
| all combined | 0.67× | +0.54 |

Bilinear resampling averages neighbouring pixels, so it both shrinks the noise
and correlates it. A model trained through that pipeline sees noise about a
third weaker and spatially smooth compared with what arrives at test time.

That **may** contribute to TinyNAFNet being the smoother of the two models, but
we did not run the controlled experiment (same architecture, two augmentation
policies), so it is stated as a hypothesis and not as a finding.

---

## 4. Why HF ratio exists

`HF ratio` is high-frequency energy above 0.25 cycles/px, relative to ground
truth. 1.0 means texture matches the target.

It exists because both scored distortion metrics **reward blur**. Applying a
Gaussian blur to the bicubic baseline:

| blur σ | PSNR | SSIM | HF energy vs GT |
|---|---|---|---|
| ground truth | — | — | 1.00 |
| 0.0 | 27.24 | 0.8314 | 1.29 |
| 1.3 | **30.84** | — | **0.45** |

Blurring buys **+3.6 dB** while cutting texture to 45% of ground truth. This is
the perception–distortion tradeoff (Blau & Michaeli, 2018), not a bug: L1 and L2
losses converge to the posterior median and mean, and when speckle makes texture
ambiguous, the mean of all plausible textures is a blur.

**This is visible in our own output, not only in the metric.** In
`val_example.png` the ground truth carries the letters **"aa"** printed on the
shelf at the top left. Our restoration recovers their position and rough shape,
but the letterforms come back soft: the stroke edges are rounded off and the
counters are filled in compared with the ground truth, so the pair reads as a
smudge rather than as two crisp characters. The information needed to resolve
them survived the speckle, and the model still returned the smooth average. That
single detail is the whole argument for tracking HF ratio: a restoration that
loses it can still post a respectable PSNR.

We could not remove the effect, but we could measure it and choose where to sit.
The LPIPS term and the `SSIM − LPIPS` selection criterion are both consequences
of this measurement. Selecting on SSIM alone while training for texture would
have picked the blurriest epoch and silently undone the experiment.

---

## 5. Restored examples

| file | what it shows |
|---|---|
| `val_example.png` | a held-out validation image: NoisyLR input, bicubic baseline, our restoration, ground truth, with a 64 px zoom |
| `test_example.png` | a genuine test input, where no ground truth exists, so the comparison is against bicubic |
| `failure_case.png` | a below-average case at 25.71 dB |

**Where the model works.** Speckle is removed cleanly across the full intensity
range, including the near-black regions that make up 52% of every image. Edges
and large structures come back sharp, and the output never invents structure
that is absent from the input.

**Where it fails.** In `failure_case.png` the fine repeated texture — the book
spines in the zoom — is recovered as smooth tone rather than as individual
lines, and the "aa" lettering discussed in section 4 loses its stroke definition.
When speckle destroys a texture completely, the loss-optimal answer is the
average of every texture that could have been there, and that average is a blur.
This is the same effect quantified in section 4, visible on one image.

**Honest limitations.**

- HF ratio is 0.50: half of ground-truth high-frequency energy is retained.
- The measured speckle level varies about 1.8× across images (p10 ≈ 0.145,
  p90 ≈ 0.26) and the model is not conditioned on it, so it applies one
  compromise denoising strength to every image. This is the largest concrete
  improvement we did not get to.
- Out-of-distribution image content is untested. Our validation split is drawn
  from the same distribution as training, so it cannot measure this.
- The ensemble LPIPS figures in section 3 rest on 3 instrumented images. The
  direction is unambiguous, the magnitude is an estimate.

---

## 6. Runtime

| | |
|---|---|
| throughput | 152 images/s |
| batch size | 32 |
| precision | fp32 |
| hardware | NVIDIA laptop GPU, 3.67 GiB usable, CUDA 12.x |
| software | Python 3.12, PyTorch 2.x, NumPy 2.x |
| timing method | `time.perf_counter()` around the entire pipeline, with an explicit `torch.cuda.synchronize()` before the device-to-host copy is timed |

The timed window covers disk read, preprocessing, CPU→GPU transfer, model
execution, GPU→CPU transfer, post-processing and saving. `run.py` prints the
same breakdown on every invocation.

Two throughput decisions, both measured:

| option | quality gain | cost | shipped |
|---|---|---|---|
| 8× dihedral test-time augmentation | +0.065 dB, +0.0024 SSIM | 8× compute | no |
| two-model ensemble at the best blend weight | +0.36 dB, +0.0035 SSIM, 2.2× worse LPIPS | 2× compute | no |
