"""
================================================================================
DATASET
Load any text corpus, tokenize, and serve training batches.
================================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TextDataset(Dataset):
    """
    Loads text, tokenizes, chunks into fixed-length sequences.
    Each sample is (input_ids, target_ids) where target is input shifted by 1.
    """

    def __init__(self, token_ids: list[int], context_length: int):
        self.context_length = context_length
        self.data = torch.tensor(token_ids, dtype=torch.long)
        # Number of complete sequences we can extract
        self.n_samples = max(0, (len(self.data) - 1) // context_length)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        end = start + self.context_length
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        return x, y


def load_text_files(paths: list[str | Path], separator: str = "\n") -> str:
    """Load and concatenate multiple text files."""
    texts = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.glob("**/*.txt")):
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
        elif p.exists():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
        else:
            print(f"WARNING: {p} not found, skipping")
    return separator.join(texts)


def create_dataloaders(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split tokens into train/val and return DataLoaders."""
    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = TextDataset(train_ids, context_length)
    val_ds = TextDataset(val_ids, context_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    print(f"Train: {len(train_ds)} samples ({len(train_ids):,} tokens)")
    print(f"Val:   {len(val_ds)} samples ({len(val_ids):,} tokens)")
    return train_loader, val_loader
