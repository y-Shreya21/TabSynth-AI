"""Pydantic v2 schemas for the TabSynth Analytics REST API."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    num_rows: int
    num_columns: int
    target_column: Optional[str] = None
    file_path: str


class TableSchemaResponse(BaseModel):
    dataset_name: str
    num_rows: int
    num_columns: int
    continuous_columns: List[str]
    categorical_columns: List[str]
    cardinalities: List[int]
    class_distribution: Optional[Dict[str, float]] = None


class TrainHyperparameters(BaseModel):
    d_model: int = 64
    transformer_layers: int = 2
    attention_heads: int = 4
    mlp_hidden_dim: int = 256
    mlp_layers: int = 3
    dropout: float = 0.1
    diffusion_steps: int = 100
    noise_schedule: str = "cosine"
    batch_size: int = 256
    learning_rate: float = 2e-4
    epochs: int = 30
    lambda_num: float = 1.0
    lambda_cat: float = 0.2
    lambda_aux: float = 0.1


class TrainRequest(BaseModel):
    data_path: str = "data/raw/adult.csv"
    target_column: Optional[str] = "income"
    model_type: str = "hybrid_transformer_mlp"
    hyperparameters: TrainHyperparameters = Field(default_factory=TrainHyperparameters)
    device: str = "auto"


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    current_epoch: int
    total_epochs: int
    latest_train_loss: Optional[float] = None
    latest_val_loss: Optional[float] = None
    best_checkpoint_path: Optional[str] = None


class GenerateRequest(BaseModel):
    checkpoint_path: str
    num_samples: int = 1000
    condition_strategy: str = "balanced"  # "balanced", "natural", "class_specific", "unconditional"
    specific_class: Optional[int] = None
    cfg_scale: float = 1.0
    steps: int = 100
    output_path: Optional[str] = "data/synthetic/generated_records.csv"


class GenerateResponse(BaseModel):
    status: str
    num_samples: int
    output_path: str
    preview_records: List[Dict[str, Any]]


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
