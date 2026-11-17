"""Service handling conditional and unconditional synthetic tabular data generation."""

import os
import uuid
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from api.config import settings
from api.schemas.generation import GenerateRequest, GenerateResponse, ConditionalGenerateRequest
from api.db.models import ArtifactRecord
from api.db.repositories import ModelRepository, ArtifactRepository
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.training.trainer import select_device
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.baselines.mlp_diffusion import TabDDPMMLPModel
from tabsynth.sampling.sampler import TabSynthSampler

logger = logging.getLogger("tabsynth.generation")


class GenerationService:
    """Manages model sampling, Classifier-Free Guidance, and synthetic export."""

    @classmethod
    def generate(cls, request: GenerateRequest) -> GenerateResponse:
        """Generate synthetic records from a trained checkpoint or registered model."""
        checkpoint_path = request.checkpoint_path
        if not checkpoint_path and request.model_id:
            model_rec = ModelRepository.get_by_id(request.model_id)
            if not model_rec:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_id}' not found.",
                )
            checkpoint_path = model_rec.checkpoint_path

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model checkpoint path '{checkpoint_path}' does not exist.",
            )

        device = select_device("auto")
        try:
            checkpoint = CheckpointManager.load(checkpoint_path=checkpoint_path, device=device)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load checkpoint: {str(exc)}",
            )

        preprocessor = checkpoint["preprocessor"]
        schema = preprocessor.schema

        num_classes = len(schema.class_distribution) if schema.class_distribution else 2
        
        # Resolve model configuration and architecture dynamically
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config") or {}
        is_hybrid = any("transformer" in k for k in state_dict.keys())

        d_model = config.get("d_model")
        if d_model is None:
            if "time_embedding.mlp.0.weight" in state_dict:
                d_model = state_dict["time_embedding.mlp.0.weight"].shape[1]
            elif "class_embedding.embedding.weight" in state_dict:
                d_model = state_dict["class_embedding.embedding.weight"].shape[1]
            else:
                d_model = 64

        if is_hybrid:
            tr_layers = config.get("transformer_layers")
            if tr_layers is None:
                tr_keys = [k for k in state_dict.keys() if "transformer.encoder.layers." in k and k.endswith(".linear1.weight")]
                tr_layers = max(1, len(tr_keys))

            mlp_layers = config.get("mlp_layers")
            if mlp_layers is None:
                mlp_keys = [k for k in state_dict.keys() if "denoiser.blocks." in k and k.endswith(".linear1.weight")]
                mlp_layers = max(1, len(mlp_keys))

            mlp_hidden_dim = config.get("mlp_hidden_dim", 256)
            if "denoiser.blocks.0.linear1.weight" in state_dict:
                mlp_hidden_dim = state_dict["denoiser.blocks.0.linear1.weight"].shape[0]

            model = TabSynthHybridModel(
                num_continuous=len(schema.continuous_columns),
                cardinalities=schema.cardinalities,
                num_classes=num_classes,
                d_model=d_model,
                transformer_layers=tr_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_layers=mlp_layers,
            )
        else:
            mlp_layers = config.get("mlp_layers")
            if mlp_layers is None:
                block_keys = [k for k in state_dict.keys() if "denoiser.blocks." in k and k.endswith(".norm.weight")]
                mlp_layers = max(1, len(block_keys))

            mlp_hidden_dim = config.get("mlp_hidden_dim")
            if mlp_hidden_dim is None:
                if "denoiser.input_proj.weight" in state_dict:
                    mlp_hidden_dim = state_dict["denoiser.input_proj.weight"].shape[0]
                else:
                    mlp_hidden_dim = 256

            model = TabDDPMMLPModel(
                num_continuous=len(schema.continuous_columns),
                cardinalities=schema.cardinalities,
                num_classes=num_classes,
                d_model=d_model,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_layers=mlp_layers,
            )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        steps = request.steps or 100
        gaussian_diff = GaussianDiffusion(num_timesteps=steps, device=device)
        multinomial_diff = MultinomialDiffusion(cardinalities=schema.cardinalities, num_timesteps=steps, device=device)

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

        # Output file destination
        out_path = request.output_path
        if not out_path:
            gen_id = f"synth_{uuid.uuid4().hex[:8]}"
            out_path = os.path.join(settings.synthetic_data_dir, f"{gen_id}.csv")
        
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        synth_df.to_csv(out_path, index=False)

        # Record artifact
        ArtifactRepository.create(
            ArtifactRecord(
                id=f"art_{uuid.uuid4().hex[:8]}",
                artifact_type="synthetic_csv",
                file_path=out_path,
            )
        )

        preview = synth_df.head(10).to_dict(orient="records")
        return GenerateResponse(
            status="completed",
            num_samples=len(synth_df),
            output_path=out_path,
            preview_records=preview,
        )

    @classmethod
    def generate_conditional(cls, request: ConditionalGenerateRequest) -> GenerateResponse:
        """Specialized class-specific conditional generation."""
        gen_req = GenerateRequest(
            checkpoint_path=request.checkpoint_path,
            model_id=request.model_id,
            num_samples=request.num_samples,
            condition_strategy="class_specific",
            specific_class=request.target_class,
            cfg_scale=request.cfg_scale,
            steps=request.steps,
            output_path=request.output_path,
        )
        return cls.generate(gen_req)
