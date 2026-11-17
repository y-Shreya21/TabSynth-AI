"""Schemas for scientific evaluation requests and benchmarks."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class FidelityEvalRequest(BaseModel):
    real_data_path: str
    synthetic_data_path: str
    target_column: Optional[str] = None


class UtilityEvalRequest(BaseModel):
    real_train_path: str
    synthetic_train_path: str
    real_test_path: str
    target_column: str


class PrivacyEvalRequest(BaseModel):
    real_train_path: str
    real_test_path: str
    synthetic_path: str
    subsample_size: int = 1000
