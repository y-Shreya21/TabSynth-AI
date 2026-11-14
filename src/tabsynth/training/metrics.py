"""Metric tracking, loss accumulation, and training telemetry."""

from typing import Dict, Any, Optional
import time


class MetricAccumulator:
    """Accumulates and computes average training and validation metrics per epoch."""

    def __init__(self, prefix: str = ""):
        self.prefix = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
        self.metrics_sum: Dict[str, float] = {}
        self.total_samples: int = 0
        self.total_batches: int = 0
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def reset(self) -> None:
        """Resets all metric accumulators and timer."""
        self.metrics_sum.clear()
        self.total_samples = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.end_time = None

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Updates accumulated metrics with values from a single mini-batch."""
        for key, val in metrics.items():
            k = f"{self.prefix}{key}" if not key.startswith(self.prefix) else key
            self.metrics_sum[k] = self.metrics_sum.get(k, 0.0) + float(val) * batch_size

        self.total_samples += batch_size
        self.total_batches += 1

    def stop_timer(self) -> float:
        """Stops the timer and returns elapsed seconds."""
        self.end_time = time.time()
        return self.end_time - self.start_time

    def compute(self) -> Dict[str, float]:
        """Computes sample-weighted average across all accumulated batches."""
        if self.total_samples == 0:
            return {}

        results = {}
        for key, total_val in self.metrics_sum.items():
            results[key] = total_val / self.total_samples

        elapsed = (self.end_time or time.time()) - self.start_time
        results[f"{self.prefix}elapsed_sec"] = elapsed
        results[f"{self.prefix}throughput_samples_per_sec"] = (
            self.total_samples / max(1e-6, elapsed)
        )
        return results

    def format_summary(self) -> str:
        """Formats the computed metrics into a readable string summary."""
        computed = self.compute()
        parts = []
        for k, v in computed.items():
            if "throughput" in k:
                parts.append(f"{k}: {v:.1f}/s")
            elif "elapsed" in k:
                parts.append(f"{k}: {v:.2f}s")
            else:
                parts.append(f"{k}: {v:.4f}")
        return " | ".join(parts)
