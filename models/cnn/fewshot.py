import torch
from collections import defaultdict
from torch.utils.data import Dataset
from models.cnn.dataset import CLASSES


class FewShotMultiLabelDataset(Dataset):
    """
    Selects a small fixed subset from a base DrumOnsetWindowDataset.

    It tries to keep at most `shots_per_class` positive examples per class.
    Because this is multi-label, one window can count for several classes.
    """

    def __init__(
        self,
        base_dataset,
        shots_per_class=10,
        max_negatives=100,
        seed=42,
    ):
        self.base = base_dataset
        self.shots_per_class = int(shots_per_class)
        self.max_negatives = int(max_negatives)

        generator = torch.Generator().manual_seed(seed)

        class_to_indices = defaultdict(list)
        negative_indices = []

        for idx in range(len(base_dataset)):
            _, y = base_dataset[idx]

            positive_classes = torch.where(y > 0.5)[0].tolist()

            if len(positive_classes) == 0:
                negative_indices.append(idx)
            else:
                for c in positive_classes:
                    class_to_indices[c].append(idx)

        selected = set()

        for c, indices in class_to_indices.items():
            if len(indices) == 0:
                continue

            perm = torch.randperm(len(indices), generator=generator).tolist()
            chosen = [indices[i] for i in perm[:shots_per_class]]
            selected.update(chosen)

        if self.max_negatives > 0 and len(negative_indices) > 0:
            perm = torch.randperm(len(negative_indices), generator=generator).tolist()
            chosen_neg = [
                negative_indices[i]
                for i in perm[: min(self.max_negatives, len(negative_indices))]
            ]
            selected.update(chosen_neg)

        self.indices = sorted(selected)
        print("Few-shot positives found per class:")
        for c, name in enumerate(CLASSES):
            available = len(class_to_indices.get(c, []))
            selected_c = min(available, shots_per_class)
            print(f"  {name:15s}: available={available:4d}, selected={selected_c:3d}")
        print(
            f"Few-shot CNN dataset: {len(self.indices)} windows | "
            f"{shots_per_class} shots/class | "
            f"{max_negatives} negatives"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]