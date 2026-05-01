import itertools
import random

import torch
from torch.utils.data import Dataset

from mmseg.registry import DATASETS, DATA_SAMPLERS


@DATASETS.register_module()
class DualStreamDataset(Dataset):
    """Return labeled(A) or unlabeled(B) sample each time."""
    def __init__(self, labeled_dataset, unlabeled_dataset, labeled_ratio=0.5, seed=0):
        self.A = DATASETS.build(labeled_dataset)
        self.B = DATASETS.build(unlabeled_dataset)
        self.len_a = len(self.A)
        self.len_b = len(self.B)

        assert 0.0 <= labeled_ratio <= 1.0
        self.labeled_ratio = labeled_ratio

        # local RNG (avoid affecting global randomness)
        self.rng = random.Random(seed)

        # define epoch length (you can choose max or sum; max is common)
        self._len = max(len(self.A), len(self.B))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            stream, sample_idx = idx
            if stream in ('labeled', 'A', 0):
                return self.A[sample_idx % self.len_a]
            if stream in ('unlabeled', 'B', 1):
                return self.B[sample_idx % self.len_b]
            raise KeyError(f'Unknown stream specifier: {stream}')

        if self.rng.random() < self.labeled_ratio:
            return self.A[idx % self.len_a]
        else:
            return self.B[idx % self.len_b]


@DATA_SAMPLERS.register_module()
class FixedRatioBatchSampler:
    def __init__(self,
                 sampler,
                 batch_size,
                 labeled_in_batch=1,
                 unlabeled_in_batch=1):
        self.sampler = sampler
        self.dataset = sampler.dataset
        self.batch_size = batch_size
        self.labeled_in_batch = labeled_in_batch
        self.unlabeled_in_batch = unlabeled_in_batch

        if labeled_in_batch + unlabeled_in_batch != batch_size:
            raise ValueError(
                'labeled_in_batch + unlabeled_in_batch must equal batch_size, '
                f'got {labeled_in_batch} + {unlabeled_in_batch} != {batch_size}')
        if not isinstance(self.dataset, DualStreamDataset):
            raise TypeError(
                'FixedRatioBatchSampler requires DualStreamDataset, '
                f'got {type(self.dataset)!r}')

        self.shuffle = getattr(sampler, 'shuffle', True)
        self.seed = getattr(sampler, 'seed', 0)
        self.rank = getattr(sampler, 'rank', 0)
        self.world_size = getattr(sampler, 'world_size', 1)

    def _infinite_indices(self, size):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(size, generator=generator).tolist()
            else:
                yield from torch.arange(size).tolist()

    def _indices_of_rank(self, size):
        yield from itertools.islice(
            self._infinite_indices(size), self.rank, None, self.world_size)

    def __iter__(self):
        labeled_iter = self._indices_of_rank(self.dataset.len_a)
        unlabeled_iter = self._indices_of_rank(self.dataset.len_b)

        while True:
            batch = []
            for _ in range(self.labeled_in_batch):
                batch.append(('labeled', next(labeled_iter)))
            for _ in range(self.unlabeled_in_batch):
                batch.append(('unlabeled', next(unlabeled_iter)))
            yield batch

    def __len__(self):
        return len(self.sampler)

