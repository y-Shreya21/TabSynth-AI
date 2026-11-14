"""Comprehensive training manager for mixed-type tabular diffusion models."""

from typing import Dict, Any, Optional, List, Union
import logging
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tabsynth.diffusion.utils import select_device
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.training.optimizer import build_optimizer
from tabsynth.training.scheduler import build_scheduler, get_current_lr
from tabsynth.training.early_stopping import EarlyStopping
from tabsynth.training.metrics import MetricAccumulator
from tabsynth.training.train_step import execute_train_step
from tabsynth.training.validation import evaluate_validation
from tabsynth.data.preprocessor import TabularPreprocessor

logger = logging.getLogger(__name__)


class TabSynthTrainer:
    """Coordinates training, validation, learning rate scheduling, early stopping, and checkpointing.
    
    Fully optimized for Apple Silicon MPS and CUDA with CPU fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        gaussian_diffusion: Optional[GaussianDiffusion] = None,
        multinomial_diffusion: Optional[CategoricalDiffusion] = None,
        mixed_diffusion: Optional[MixedTypeDiffusion] = None,
        loss_fn: Optional[TabSynthLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: Union[str, torch.device] = "auto",
        checkpoint_dir: str = "checkpoints/",
        keep_top_k: int = 2,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.preprocessor = preprocessor
        self.max_grad_norm = max_grad_norm
        self.config = config or {}

        # Resolve cardinalities from preprocessor or diffusion engine
        self.cardinalities: List[int] = []
        if preprocessor is not None and hasattr(preprocessor, "schema") and preprocessor.schema is not None:
            self.cardinalities = preprocessor.schema.cardinalities
        elif multinomial_diffusion is not None and hasattr(multinomial_diffusion, "cardinalities"):
            self.cardinalities = multinomial_diffusion.cardinalities
        elif mixed_diffusion is not None and hasattr(mixed_diffusion, "categorical_cardinalities"):
            self.cardinalities = mixed_diffusion.categorical_cardinalities

        # Transfer diffusion modules to target device
        self.gaussian_diff = gaussian_diffusion
        if self.gaussian_diff is not None and hasattr(self.gaussian_diff, "to_device"):
            self.gaussian_diff.to_device(self.device)

        self.multinomial_diff = multinomial_diffusion
        if self.multinomial_diff is not None and hasattr(self.multinomial_diff, "to_device"):
            self.multinomial_diff.to_device(self.device)

        self.mixed_diff = mixed_diffusion
        if self.mixed_diff is not None and hasattr(self.mixed_diff, "to_device"):
            self.mixed_diff.to_device(self.device)

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = TabSynthLoss(
                multinomial_diffusion=self.multinomial_diff,
                cardinalities=self.cardinalities,
            ).to(self.device)

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = build_optimizer(
                model=self.model,
                optimizer_type="adamw",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

        # Learning rate scheduler
        self.scheduler = scheduler

        # Early stopping monitor
        self.early_stopping = early_stopping

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir, keep_top_k=keep_top_k)
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int, max_grad_norm: Optional[float] = None) -> Dict[str, float]:
        """Runs a single training epoch across all mini-batches."""
        self.model.train()
        grad_norm_limit = max_grad_norm if max_grad_norm is not None else self.max_grad_norm
        accumulator = MetricAccumulator(prefix="")

        for batch in self.train_loader:
            x_num_0, x_cat_0_list, _ = batch
            batch_size = x_num_0.shape[0] if x_num_0 is not None and x_num_0.ndim >= 2 and x_num_0.shape[0] > 0 else len(x_cat_0_list[0])

            step_metrics = execute_train_step(
                model=self.model,
                optimizer=self.optimizer,
                batch=batch,
                loss_fn=self.loss_fn,
                gaussian_diffusion=self.gaussian_diff,
                multinomial_diffusion=self.multinomial_diff,
                mixed_diffusion=self.mixed_diff,
                device=self.device,
                max_grad_norm=grad_norm_limit,
                cardinalities=self.cardinalities,
            )

            accumulator.update(step_metrics, batch_size=batch_size)

        accumulator.stop_timer()
        epoch_metrics = accumulator.compute()

        # Step learning rate scheduler if stepped per epoch
        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        # Apple Silicon MPS memory maintenance
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluates loss over the validation partition."""
        if self.val_loader is None:
            return {}

        return evaluate_validation(
            model=self.model,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            gaussian_diffusion=self.gaussian_diff,
            multinomial_diffusion=self.multinomial_diff,
            mixed_diffusion=self.mixed_diff,
            device=self.device,
            cardinalities=self.cardinalities,
        )

    def fit(
        self,
        epochs: int = 50,
        verbose_interval: int = 10,
        prefix: str = "tabsynth",
    ) -> List[Dict[str, Any]]:
        """Executes full multi-epoch training, validation, early stopping, and checkpointing.
        
        Args:
            epochs: Total epochs to train.
            verbose_interval: Logging frequency in epochs.
            prefix: Prefix for saved checkpoint filenames.
            
        Returns:
            List of training history dictionaries per epoch.
        """
        best_val_loss = float("inf")
        logger.info(f"Starting training on device: {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch=epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            current_lr = get_current_lr(self.optimizer)
            val_total = val_metrics.get("val_loss_total", train_metrics.get("loss_total", 0.0))

            is_best = val_total < best_val_loss
            if is_best:
                best_val_loss = val_total
                if self.preprocessor is not None:
                    self.checkpoint_manager.save(
                        model=self.model,
                        preprocessor=self.preprocessor,
                        epoch=epoch,
                        val_loss=val_total,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=self.config,
                        prefix=prefix,
                    )

            # Check early stopping
            should_terminate = False
            if self.early_stopping is not None and self.val_loader is not None:
                should_terminate = self.early_stopping(
                    current_loss=val_total,
                    epoch=epoch,
                    model=self.model,
                )

            record = {
                "epoch": epoch,
                "elapsed_sec": elapsed,
                "learning_rate": current_lr,
                **train_metrics,
                **val_metrics,
                "is_best": is_best,
            }
            self.history.append(record)

            # Logging
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs or is_best or should_terminate:
                val_str = f" | Val Loss: {val_total:.4f}" if self.val_loader is not None else ""
                best_marker = " [*BEST*]" if is_best else ""
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_metrics.get('loss_total', 0.0):.4f}"
                    f"{val_str} | "
                    f"LR: {current_lr:.2e} | "
                    f"{elapsed:.2f}s{best_marker}"
                )

            if should_terminate:
                print(f"[*] Early stopping condition met at epoch {epoch}. Terminating training.")
                break

        return self.history


# Class alias for nomenclature flexibility
TabularTrainer = TabSynthTrainer
