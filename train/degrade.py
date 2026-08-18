import numpy as np

# Fitted from paired data with fit_degradation.py:
#   y = box_downsample_2x(x) * (1 + N(0, s^2)) + N(0, g^2)
# Speckle is Gaussian-multiplicative (kurtosis 3.03; uniform rejected p=1e-20).
SPECKLE_STD = 0.163
GAUSS_STD = 0.004


def box_downsample(x):
    h, w = x.shape[-2:]
    return x.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


class Degrader:
    def __init__(self, speckle_std=SPECKLE_STD, gauss_std=GAUSS_STD,
                 jitter=0.2, seed=None):
        self.speckle_std = speckle_std
        self.gauss_std = gauss_std
        self.jitter = jitter          # per-image level jitter, for blind robustness
        self.rng = np.random.default_rng(seed)

    def __call__(self, gt_hr):
        """(2H, 2W) float32 in [0,1] -> (H, W) float32 noisy."""
        j = self.jitter
        s = self.speckle_std * (1 + self.rng.uniform(-j, j)) if j else self.speckle_std
        g = self.gauss_std * (1 + self.rng.uniform(-j, j)) if j else self.gauss_std
        lr = box_downsample(gt_hr.astype(np.float32))
        out = lr * (1.0 + s * self.rng.standard_normal(lr.shape).astype(np.float32))
        if g > 0:
            out = out + g * self.rng.standard_normal(lr.shape).astype(np.float32)
        return out.astype(np.float32)