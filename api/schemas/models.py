"""Schemas for model definition, training requests, and metadata."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TrainHyperparameters(BaseModel):
    d_model: int = Field(default=64, gt=0)
    transformer_layers: int = Field(default=2, ge=0)
    attention_heads: int = Field(default=4, gt=0)
    mlp_hidden_dim: int = Field(default=256, gt=0)
    mlp_layers: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    diffusion_steps: int = Field(default=100, gt=0)
    noise_schedule: str = "cosine"
    batch_size: int = Field(default=256, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    epochs: int = Field(default=30, gt=0)
    lambda_num: float = Field(default=1.0, ge=0.0)
    lambda_cat: float = Field(default=0.2, ge=0.0)
    lambda_aux: float = Field(default=0.1, ge=0.0)
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
