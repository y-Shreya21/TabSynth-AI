"""Schemas for model definition, training requests, and metadata."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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
    seed: int = 42


class TrainRequest(BaseModel):
    data_path: Optional[str] = None
    dataset_id: Optional[str] = None
    model_name: Optional[str] = None
    target_column: Optional[str] = "income"
    model_type: str = "hybrid_transformer_mlp"  # "hybrid_transformer_mlp" or "mlp_baseline"
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


class ModelDetailResponse(BaseModel):
    id: str
    name: str
    model_type: str
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    checkpoint_path: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelDetailResponse]
