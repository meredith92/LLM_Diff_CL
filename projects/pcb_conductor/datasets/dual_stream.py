import random
from torch.utils.data import Dataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class DualStreamDataset(Dataset):
    """Return labeled(A) or unlabeled(B) sample each time."""
    def __init__(self, labeled_dataset, unlabeled_dataset, labeled_ratio=0.5, seed=0):
        self.A = DATASETS.build(labeled_dataset)
        self.B = DATASETS.build(unlabeled_dataset)

        assert 0.0 <= labeled_ratio <= 1.0
        self.labeled_ratio = labeled_ratio

        # local RNG (avoid affecting global randomness)
        self.rng = random.Random(seed)

        # define epoch length (you can choose max or sum; max is common)
        self._len = max(len(self.A), len(self.B))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.rng.random() < self.labeled_ratio:
            return self.A[idx % len(self.A)]
        else:
            return self.B[idx % len(self.B)]

