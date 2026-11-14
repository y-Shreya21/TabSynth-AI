"""Model checkpointing and state restoration with top-k storage management."""

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import glob
import shutil
import logging
import torch
import torch.nn as nn
from tabsynth.data.preprocessor import TabularPreprocessor

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages serializing and restoring models, optimizers, schedulers, and preprocessors."""

    def __init__(
        self,
        save_dir: str = "checkpoints/",
        keep_top_k: int = 2,
    ):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)
        self.best_checkpoints: List[Tuple[float, str]] = []

    def save(
        self,
        model: nn.Module,
        preprocessor: TabularPreprocessor,
        epoch: int,
        val_loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        prefix: str = "tabsynth",
    ) -> str:
        """Saves a checkpoint bundle and prunes older checkpoints exceeding top_k."""
        checkpoint_name = f"{prefix}_epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        payload = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "model_state_dict": model.state_dict(),
            "preprocessor": preprocessor,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "metadata": metadata,
        }

        torch.save(payload, checkpoint_path)

        # Track and prune top-k
        self.best_checkpoints.append((val_loss, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by ascending loss (lower is better)

        # If current checkpoint is the overall best, create a standardized copy 'best_checkpoint.pt'
        if self.best_checkpoints and self.best_checkpoints[0][1] == checkpoint_path:
            best_link_path = os.path.join(self.save_dir, f"{prefix}_best.pt")
            try:
                shutil.copyfile(checkpoint_path, best_link_path)
            except OSError as err:
                logger.warning(f"Could not copy best checkpoint link: {err}")

        # Prune worst checkpoints exceeding top_k
        while len(self.best_checkpoints) > self.keep_top_k:
            _, worst_path = self.best_checkpoints.pop(-1)
            if os.path.exists(worst_path) and worst_path != checkpoint_path:
                try:
                    os.remove(worst_path)
                    logger.debug(f"Pruned checkpoint: {worst_path}")
                except OSError:
                    pass

        return checkpoint_path

    def get_best_checkpoint(self, prefix: str = "tabsynth") -> Optional[str]:
        """Returns the path to the best recorded checkpoint."""
        best_standard = os.path.join(self.save_dir, f"{prefix}_best.pt")
        if os.path.exists(best_standard):
            return best_standard
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None

    @staticmethod
    def load(
        checkpoint_path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Loads checkpoint dictionary and optionally populates model/optimizer weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        target_device = torch.device(device) if isinstance(device, str) else device
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(target_device)

        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
