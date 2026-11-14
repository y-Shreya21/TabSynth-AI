"""FastAPI routes for dataset management, model training, generation, and evaluation."""

import os
import uuid
import threading
import pandas as pd
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from api.schemas import (
    DatasetUploadResponse,
    TableSchemaResponse,
    TrainRequest,
    TrainStatusResponse,
    GenerateRequest,
    GenerateResponse,
    FidelityEvalRequest,
    UtilityEvalRequest,
    PrivacyEvalRequest,
)
from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.models.embeddings import SinusoidalTimeEmbedding
from tabsynth.baselines.mlp_diffusion import TabDDPMMLPModel
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.trainer import TabSynthTrainer, select_device
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.sampling.sampler import TabSynthSampler
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy

router = APIRouter()

# In-memory tracking of background training jobs
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


def run_training_job(job_id: str, request: TrainRequest):
    """Background worker for asynchronous model training."""
    try:
        TRAINING_JOBS[job_id]["status"] = "running"
        df = pd.read_csv(request.data_path)
        target_col = request.target_column

        df_train, df_val, df_test = create_stratified_split(df, target_column=target_col, seed=42)
        schema = infer_table_schema(df_train, target_column=target_col)
        preprocessor = TabularPreprocessor(schema=schema, seed=42).fit(df_train)

        train_loader, val_loader, _ = build_dataloaders(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            preprocessor=preprocessor,
            batch_size=request.hyperparameters.batch_size,
        )

        device = select_device(request.device)
        hp = request.hyperparameters

        gaussian_diff = GaussianDiffusion(num_timesteps=hp.diffusion_steps, schedule_type=hp.noise_schedule, device=device)
        multinomial_diff = MultinomialDiffusion(cardinalities=schema.cardinalities, num_timesteps=hp.diffusion_steps, device=device)

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

        ckpt_dir = f"checkpoints/{job_id}"
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
            checkpoint_dir=ckpt_dir,
        )

        for ep in range(1, hp.epochs + 1):
            t_losses = trainer.train_epoch(epoch=ep)
            v_losses = trainer.validate()
            val_loss = v_losses["val_loss_total"]

            TRAINING_JOBS[job_id]["current_epoch"] = ep
            TRAINING_JOBS[job_id]["latest_train_loss"] = float(t_losses["loss_total"])
            TRAINING_JOBS[job_id]["latest_val_loss"] = float(val_loss)

            saved_path = trainer.checkpoint_manager.save(
                model=model,
                preprocessor=preprocessor,
                epoch=ep,
                val_loss=val_loss,
            )
            TRAINING_JOBS[job_id]["best_checkpoint_path"] = saved_path

        TRAINING_JOBS[job_id]["status"] = "completed"

    except Exception as e:
        TRAINING_JOBS[job_id]["status"] = "failed"
        TRAINING_JOBS[job_id]["error"] = str(e)


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...), name: str = Form("uploaded_dataset")):
    os.makedirs("data/raw", exist_ok=True)
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    file_path = f"data/raw/{dataset_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    df = pd.read_csv(file_path)
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=name,
        num_rows=len(df),
        num_columns=len(df.columns),
        file_path=file_path,
    )


@router.get("/datasets/schema", response_model=TableSchemaResponse)
async def get_dataset_schema(data_path: str = "data/raw/adult.csv", target_column: str = "income"):
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    schema = infer_table_schema(df, target_column=target_column)

    return TableSchemaResponse(
        dataset_name=schema.dataset_name,
        num_rows=schema.num_rows,
        num_columns=schema.num_columns,
        continuous_columns=schema.continuous_columns,
        categorical_columns=schema.categorical_columns,
        cardinalities=schema.cardinalities,
        class_distribution=schema.class_distribution,
    )


@router.post("/models/train", response_model=Dict[str, str])
async def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    TRAINING_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "current_epoch": 0,
        "total_epochs": request.hyperparameters.epochs,
        "latest_train_loss": None,
        "latest_val_loss": None,
        "best_checkpoint_path": None,
    }

    background_tasks.add_task(run_training_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@router.get("/models/status", response_model=TrainStatusResponse)
async def get_training_status(job_id: str):
    if job_id not in TRAINING_JOBS:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    info = TRAINING_JOBS[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=info["status"],
        current_epoch=info["current_epoch"],
        total_epochs=info["total_epochs"],
        latest_train_loss=info.get("latest_train_loss"),
        latest_val_loss=info.get("latest_val_loss"),
        best_checkpoint_path=info.get("best_checkpoint_path"),
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found at {request.checkpoint_path}")

    device = select_device("auto")
    checkpoint = CheckpointManager.load(checkpoint_path=request.checkpoint_path, device=device)
    preprocessor = checkpoint["preprocessor"]
    schema = preprocessor.schema

    num_classes = len(schema.class_distribution) if schema.class_distribution else 2
    model = TabSynthHybridModel(
        num_continuous=len(schema.continuous_columns),
        cardinalities=schema.cardinalities,
        num_classes=num_classes,
        d_model=64,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    gaussian_diff = GaussianDiffusion(num_timesteps=request.steps, device=device)
    multinomial_diff = MultinomialDiffusion(cardinalities=schema.cardinalities, num_timesteps=request.steps, device=device)

    sampler = TabSynthSampler(
        model=model,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        preprocessor=preprocessor,
        device=device,
    )

    synth_df = sampler.sample(
        num_samples=request.num_samples,
        condition_strategy=request.condition_strategy,
        specific_class=request.specific_class,
        cfg_scale=request.cfg_scale,
    )

    os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
    synth_df.to_csv(request.output_path, index=False)

    preview = synth_df.head(5).to_dict(orient="records")
    return GenerateResponse(
        status="completed",
        num_samples=len(synth_df),
        output_path=request.output_path,
        preview_records=preview,
    )


@router.post("/evaluate/fidelity")
async def evaluate_fidelity_endpoint(request: FidelityEvalRequest):
    if not os.path.exists(request.real_data_path) or not os.path.exists(request.synthetic_data_path):
        raise HTTPException(status_code=404, detail="Real or synthetic data file not found")

    df_real = pd.read_csv(request.real_data_path)
    df_synth = pd.read_csv(request.synthetic_data_path)
    schema = infer_table_schema(df_real, target_column=request.target_column)

    return evaluate_statistical_fidelity(
        real_df=df_real,
        synth_df=df_synth,
        continuous_cols=schema.continuous_columns,
        categorical_cols=schema.categorical_columns,
    )


@router.post("/evaluate/utility")
async def evaluate_utility_endpoint(request: UtilityEvalRequest):
    df_train_real = pd.read_csv(request.real_train_path)
    df_train_synth = pd.read_csv(request.synthetic_train_path)
    df_test_real = pd.read_csv(request.real_test_path)

    return evaluate_downstream_utility(
        df_train_real=df_train_real,
        df_train_synth=df_train_synth,
        df_test_real=df_test_real,
        target_column=request.target_column,
    )


@router.post("/evaluate/privacy")
async def evaluate_privacy_endpoint(request: PrivacyEvalRequest):
    df_train = pd.read_csv(request.real_train_path)
    df_test = pd.read_csv(request.real_test_path)
    df_synth = pd.read_csv(request.synthetic_path)

    schema = infer_table_schema(df_train)
    return evaluate_empirical_privacy(
        df_train=df_train,
        df_test=df_test,
        df_synth=df_synth,
        continuous_cols=schema.continuous_columns,
        categorical_cols=schema.categorical_columns,
        subsample_size=request.subsample_size,
    )


@router.get("/experiments")
async def list_experiments():
    os.makedirs("experiments/results", exist_ok=True)
    exp_files = [f for f in os.listdir("experiments/results") if f.endswith(".json")]
    return {"experiments": sorted(exp_files)}
