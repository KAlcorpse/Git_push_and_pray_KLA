# Final submission checklist

| # | item | where it is satisfied |
|---|---|---|
| 1 | Mandatory solution PPT/PPTX included | `solution_presentation.pdf` (and `.pptx`) at the repo root |
| 2 | GitHub repository link accessible | verify the link in an incognito window before submitting |
| 3 | Only the three official degradations treated as benchmark requirements | README §7; the fitted chain is `y = ↓₂(x)·(1+N(0,s²)) + N(0,g²)` |
| 4 | NoisyLR values outside [0,1] handled intentionally | README §7; `run.py` passes the input through unmodified and clips only the output |
| 5 | Inference script accepts input and output directory arguments | `python run.py <input-dir> <output-dir>` |
| 6 | Training script reproduces the submitted checkpoint | `train/train.py`, exact command in README §5 |
| 7 | Model weights/config and environment specification included | `models/best.pth`, `requirements.txt` with pinned versions |
| 8 | README commands run without manual source-code edits | paths resolve relative to `run.py`; nothing to configure |
| 9 | PSNR, SSIM and LPIPS reported | README §3, `results/metrics.md` §1–2 |
| 10 | Numerical metrics and restored-image examples shown | `results/metrics.md` and the three PNGs beside it |
| 11 | End-to-end runtime, hardware, batch size, timing method stated | README §4; `run.py` prints the same breakdown every run |
| 12 | At least one baseline and one failure case | bicubic baseline in every table; `results/failure_case.png` |
| 13 | External data/models with links and licences | README §8 |
| 14 | No confidential, unlicensed or inaccessible data | README §7, last bullet |
| 15 | Dry-run in a clean environment | procedure below |

## Clean-environment dry run

Run this on a machine that has never seen the project, with the network off
after the install step.

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# disconnect the network here

python run.py /path/to/test/NoisyLR /tmp/restored
```

Then confirm the output contract:

```bash
python - <<'PY'
import numpy as np, os, sys
inp, out = sys.argv[1], sys.argv[2]
ins  = sorted(f for f in os.listdir(inp) if f.endswith(".npy"))
outs = sorted(f for f in os.listdir(out) if f.endswith(".npy"))
assert ins == outs, "filenames do not match one to one"
for f in outs:
    a, b = np.load(os.path.join(out, f)), np.load(os.path.join(inp, f))
    b = b[..., 0] if b.ndim == 3 and b.shape[-1] == 1 else (b[0] if b.ndim == 3 else b)
    assert a.ndim == 2 and a.dtype == np.float32, f
    assert a.shape == (b.shape[0]*2, b.shape[1]*2), f
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, f
print(f"{len(outs)} files: shapes, dtype, range and finiteness all pass")
PY
```

Invoke it as `python check.py /path/to/test/NoisyLR /tmp/restored`.
