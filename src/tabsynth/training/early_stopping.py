"""Early stopping monitor for model training."""

from typing import Optional, Dict, Any
import copy
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors validation loss to terminate training early and prevent overfitting."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.should_stop: bool = False
        self.best_weights: Optional[Dict[str, Any]] = None

        if self.mode not in ("min", "max"):
            raise ValueError(f"Unsupported mode: {mode}. Must be 'min' or 'max'.")

    def __call__(self, current_loss: float, epoch: int, model: Optional[nn.Module] = None) -> bool:
        """Evaluates whether current loss is an improvement.
        
        Returns:
            True if training should stop, False otherwise.
        """
        score = -current_loss if self.mode == "min" else current_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights and model is not None:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best loss was {abs(self.best_score):.4f} at epoch {self.best_epoch}."
                )
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights.")
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = copy.deepcopy(model.state_dict())

        return False

    def reset(self) -> None:
        """Resets the early stopping monitor state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
        self.best_weights = None
