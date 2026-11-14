"""PyTorch Dataset and stratified partition utilities for tabular diffusion."""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tabsynth.data.preprocessor import TabularPreprocessor


class TabularDataset(Dataset):
    """PyTorch Dataset encapsulating continuous, discrete, and conditioning tensors."""

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat_list: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.num_samples = len(x_num) if len(x_num) > 0 else len(x_cat_list[0])

        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat_list = [torch.tensor(cat, dtype=torch.float32) for cat in x_cat_list]

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.zeros(self.num_samples, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        return (
            self.x_num[idx],
            [cat[idx] for cat in self.x_cat_list],
            self.labels[idx],
        )


def collate_tabular_batch(batch):
    """Collate individual samples into batched tensors."""
    x_nums, x_cat_lists, labels = zip(*batch)

    batched_x_num = torch.stack(x_nums, dim=0)
    batched_labels = torch.stack(labels, dim=0)

    # Stack categorical one-hots column-wise
    num_cats = len(x_cat_lists[0])
    batched_cats = [
        torch.stack([sample[col_idx] for sample in x_cat_lists], dim=0)
        for col_idx in range(num_cats)
    ]

    return batched_x_num, batched_cats, batched_labels


def create_stratified_split(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions a DataFrame into train, validation, and test sets.
    
    Uses stratified sampling on the target column if provided.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0."

    stratify_target = df[target_column] if target_column in df.columns else None

    # First split: separate training from held-out (val + test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_val_test = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=stratify_target,
        random_state=seed,
    )

    # Second split: separate validation from test
    test_sub_ratio = test_ratio / val_test_ratio
    stratify_val_test = df_val_test[target_column] if target_column in df.columns else None
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=test_sub_ratio,
        stratify=stratify_val_test,
        random_state=seed,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def build_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Transforms partitions using the preprocessor and constructs PyTorch DataLoaders."""
    # Transform partitions
    x_num_tr, x_cat_tr, y_tr = preprocessor.transform(df_train)
    x_num_va, x_cat_va, y_va = preprocessor.transform(df_val)
    x_num_te, x_cat_te, y_te = preprocessor.transform(df_test)

    train_ds = TabularDataset(x_num_tr, x_cat_tr, y_tr)
    val_ds = TabularDataset(x_num_va, x_cat_va, y_va)
    test_ds = TabularDataset(x_num_te, x_cat_te, y_te)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_tabular_batch,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tabular_batch,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tabular_batch,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
