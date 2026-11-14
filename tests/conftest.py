"""TabSynth Pytest Configuration and Common Fixtures."""
import pytest
import torch
import numpy as np


@pytest.fixture
def mock_tabular_batch():
    """Generates synthetic continuous and one-hot categorical tensors for testing."""
    batch_size = 16
    num_continuous = 4
    cat_cardinalities = [3, 5, 2]
    total_cat_dim = sum(cat_cardinalities)

    x_num = torch.randn(batch_size, num_continuous, dtype=torch.float32)
    
    # Generate valid one-hot vectors for categoricals
    cat_tensors = []
    for k in cat_cardinalities:
        labels = torch.randint(0, k, (batch_size,))
        one_hot = torch.nn.functional.one_hot(labels, num_classes=k).float()
        cat_tensors.append(one_hot)
    x_cat = torch.cat(cat_tensors, dim=-1)

    labels = torch.randint(0, 2, (batch_size,))
    timesteps = torch.randint(1, 1001, (batch_size,))

    return {
        "x_num": x_num,
        "x_cat": x_cat,
        "labels": labels,
        "timesteps": timesteps,
        "batch_size": batch_size,
        "num_continuous": num_continuous,
        "cat_cardinalities": cat_cardinalities,
    }
