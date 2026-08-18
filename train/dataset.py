import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from degrade import Degrader

SCALE = 2


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, split="train", val_split=0.1, seed=42,
                 crop=64, augment=True, cache=True, synth_prob=0.0, synth_jitter=0.2):
        self.gt_dir, self.noisy_dir, self.split = gt_dir, noisy_dir, split
        self.crop = crop
        self.augment = augment and split == "train"

        names = sorted(set(os.listdir(gt_dir)) & set(os.listdir(noisy_dir)))
        names = [f for f in names if f.endswith(".npy")]
        if not names:
            raise RuntimeError(f"no overlapping .npy files in {gt_dir} and {noisy_dir}")
        random.Random(seed).shuffle(names)
        cut = int(len(names) * (1 - val_split))
        self.fnames = names[:cut] if split == "train" else names[cut:]

        self.synth_prob = synth_prob if split == "train" else 0.0
        self._degrader = Degrader(jitter=synth_jitter, seed=seed) if self.synth_prob else None
        self._rng_ready = False

        self.cache = cache
        if cache:
            g0 = np.load(os.path.join(gt_dir, self.fnames[0]))
            n0 = np.load(os.path.join(noisy_dir, self.fnames[0]))
            if g0.shape[0] != n0.shape[0] * SCALE:
                raise RuntimeError(f"expected GT = {SCALE}x NoisyLR, got {g0.shape} vs {n0.shape}")
            self._gt = np.empty((len(self.fnames), *g0.shape), np.float16)
            self._noisy = np.empty((len(self.fnames), *n0.shape), np.float16)
            for i, f in enumerate(self.fnames):
                self._gt[i] = np.load(os.path.join(gt_dir, f))
                self._noisy[i] = np.load(os.path.join(noisy_dir, f))
            print(f"[{split}] {len(self.fnames)} pairs cached, "
                  f"{(self._gt.nbytes + self._noisy.nbytes)/1e6:.0f} MB")

    def set_crop(self, crop):
        self.crop = crop

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if self.cache:
            noisy = self._noisy[idx].astype(np.float32)
            gt = self._gt[idx].astype(np.float32)
        else:
            f = self.fnames[idx]
            noisy = np.load(os.path.join(self.noisy_dir, f)).astype(np.float32)
            gt = np.load(os.path.join(self.gt_dir, f)).astype(np.float32)

        c = self.crop
        if c and self.split == "train" and c < noisy.shape[-1]:
            y = random.randint(0, noisy.shape[-2] - c)
            x = random.randint(0, noisy.shape[-1] - c)
            noisy = noisy[y:y + c, x:x + c]
            gt = gt[y * SCALE:(y + c) * SCALE, x * SCALE:(x + c) * SCALE]

        if self._degrader is not None and random.random() < self.synth_prob:
            if not self._rng_ready:  # workers fork this object; reseed per worker
                self._degrader.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
                self._rng_ready = True
            noisy = self._degrader(gt)

        noisy = torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0)
        gt = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)

        if self.augment:
            k = random.randint(0, 3)
            if k:
                noisy, gt = torch.rot90(noisy, k, (-2, -1)), torch.rot90(gt, k, (-2, -1))
            if random.random() < 0.5:
                noisy, gt = torch.flip(noisy, (-1,)), torch.flip(gt, (-1,))
            if random.random() < 0.5:
                noisy, gt = torch.flip(noisy, (-2,)), torch.flip(gt, (-2,))

        return noisy, gt


class TestDataset(Dataset):
    def __init__(self, noisy_dir):
        self.noisy_dir = noisy_dir
        self.fnames = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".npy"))
        if not self.fnames:
            raise RuntimeError(f"no .npy files in {noisy_dir}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        f = self.fnames[idx]
        arr = np.load(os.path.join(self.noisy_dir, f)).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0), f