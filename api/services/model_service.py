"""Service coordinating model training pipelines, checkpointing, and lifecycle management."""

import os
import uuid
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from api.config import settings
from api.schemas.models import TrainRequest, TrainHyperparameters
from api.db.models import ModelRecord, ArtifactRecord
from api.db.repositories import ModelRepository, DatasetRepository, ArtifactRepository
from api.services.job_service import job_manager

from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.baselines.mlp_diffusion import TabDDPMMLPModel
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.trainer import TabSynthTrainer, select_device

logger = logging.getLogger("tabsynth.models")


class ModelService:
    """Manages deep learning model training jobs, architectures, and checkpoints."""

    @classmethod
    def start_training_job(cls, request: TrainRequest) -> str:
        """Resolve dataset, validate request, and enqueue training job."""
        # Resolve data path
        data_path = request.data_path
        dataset_id = request.dataset_id
        if dataset_id:
            ds = DatasetRepository.get_by_id(dataset_id)
            if not ds:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset '{dataset_id}' not found.",
                )
            data_path = ds.file_path
            if not request.target_column and ds.target_column:
                request.target_column = ds.target_column

        if not data_path or not os.path.exists(data_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training data path '{data_path}' does not exist.",
            )

        total_epochs = request.hyperparameters.epochs
        job_id = job_manager.submit_job(
            job_type="training",
            task_fn=cls._run_training_worker,
            metadata={
                "data_path": data_path,
                "dataset_id": dataset_id,
                "target_column": request.target_column,
                "model_type": request.model_type,
                "model_name": request.model_name or f"{request.model_type}_run",
                "hyperparameters": request.hyperparameters.model_dump(),
                "device": request.device,
            },
            total_steps=total_epochs,
            request=request,
            data_path=data_path,
        )
        return job_id

    @classmethod
    def _run_training_worker(
        cls,
        request: TrainRequest,
        data_path: str,
        job_id: str,
        progress_callback,
        is_cancelled_fn,
    ) -> Dict[str, Any]:
        """Execute the PyTorch training loop inside a managed background thread."""
        hp = request.hyperparameters
        df = pd.read_csv(data_path)
        target_col = request.target_column

        df_train, df_val, df_test = create_stratified_split(df, target_column=target_col, seed=hp.seed)
        schema = infer_table_schema(df_train, target_column=target_col)
        preprocessor = TabularPreprocessor(schema=schema, seed=hp.seed).fit(df_train)

        train_loader, val_loader, _ = build_dataloaders(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            preprocessor=preprocessor,
            batch_size=hp.batch_size,
        )

        device = select_device(request.device)
        gaussian_diff = GaussianDiffusion(
            num_timesteps=hp.diffusion_steps,
            schedule_type=hp.noise_schedule,
            device=device,
        )
        multinomial_diff = MultinomialDiffusion(
            cardinalities=schema.cardinalities,
            num_timesteps=hp.diffusion_steps,
            device=device,
        )

        num_classes = len(schema.class_distribution) if schema.class_distribution else 2

        if request.model_type == "hybrid_transformer_mlp":
            model = TabSynthHybridModel(
                num_continuous=len(schema.continuous_columns),
                cardinalities=schema.cardinalities,
                num_classes=num_classes,
                d_model=hp.d_model,
                transformer_layers=hp.transformer_layers,
                attention_heads=hp.attention_heads,
                mlp_hidden_dim=hp.mlp_hidden_dim,
                mlp_layers=hp.mlp_layers,
                dropout=hp.dropout,
            )
        else:
            model = TabDDPMMLPModel(
                num_continuous=len(schema.continuous_columns),
                cardinalities=schema.cardinalities,
                num_classes=num_classes,
                d_model=hp.d_model,
                mlp_hidden_dim=hp.mlp_hidden_dim,
                mlp_layers=hp.mlp_layers,
            )

        loss_fn = TabSynthLoss(
            multinomial_diffusion=multinomial_diff,
            cardinalities=schema.cardinalities,
            lambda_num=hp.lambda_num,
            lambda_cat=hp.lambda_cat,
            lambda_aux=hp.lambda_aux,
        )

        checkpoint_dir = os.path.join(settings.checkpoint_dir, job_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        trainer = TabSynthTrainer(
            model=model,
            gaussian_diffusion=gaussian_diff,
            multinomial_diffusion=multinomial_diff,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            preprocessor=preprocessor,
            learning_rate=hp.learning_rate,
            device=str(device),
            checkpoint_dir=checkpoint_dir,
        )

        best_checkpoint = None
        best_val_loss = float("inf")

        for epoch in range(1, hp.epochs + 1):
            if is_cancelled_fn():
                logger.info(f"Training job {job_id} cancelled at epoch {epoch}.")
                break

            t_losses = trainer.train_epoch(epoch=epoch)
            v_losses = trainer.validate()
            val_loss = float(v_losses["val_loss_total"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            saved_path = trainer.checkpoint_manager.save(
                model=model,
                preprocessor=preprocessor,
                epoch=epoch,
                val_loss=val_loss,
                config=hp.model_dump(),
            )
            best_checkpoint = saved_path

            progress_callback(
                current_step=epoch,
                extra_result={
                    "current_epoch": epoch,
                    "train_loss": float(t_losses["loss_total"]),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "checkpoint_path": best_checkpoint,
                },
            )

        # Register completed model
        model_id = f"model_{uuid.uuid4().hex[:10]}"
        model_name = request.model_name or f"{request.model_type}_{job_id}"
        model_record = ModelRecord(
            id=model_id,
            name=model_name,
            model_type=request.model_type,
            dataset_id=request.dataset_id,
            target_column=target_col,
            checkpoint_path=best_checkpoint or os.path.join(checkpoint_dir, "model_best.pt"),
            hyperparameters=hp.model_dump(),
            metrics={"best_val_loss": best_val_loss, "final_epoch": epoch},
        )
        ModelRepository.create(model_record)

        if best_checkpoint:
            ArtifactRepository.create(
                ArtifactRecord(
                    id=f"art_{uuid.uuid4().hex[:8]}",
                    job_id=job_id,
                    artifact_type="checkpoint",
                    file_path=best_checkpoint,
                )
            )

        return {
            "model_id": model_id,
            "model_name": model_name,
            "checkpoint_path": model_record.checkpoint_path,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch,
        }

    @staticmethod
    def get_model(model_id: str) -> ModelRecord:
        record = ModelRepository.get_by_id(model_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found.",
            )
        return record

    @staticmethod
    def list_models() -> List[ModelRecord]:
        return ModelRepository.list_all()
