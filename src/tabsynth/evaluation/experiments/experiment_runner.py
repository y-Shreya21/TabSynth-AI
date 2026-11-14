"""Experiment runner for coordinating smoke tests, benchmarks, and ablations."""

import os
import json
import time
from typing import Dict, List, Any, Optional
import pandas as pd

from tabsynth.data.schema import TableSchema, infer_table_schema
from tabsynth.evaluation.benchmarks.baseline_registry import get_synthesizer
from tabsynth.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from .ablations import (
    run_e1_architecture_ablation,
    run_e5_class_sampling_ablation,
    run_e6_minority_augmentation_study,
    run_e7_fidelity_privacy_tradeoff,
)


class ExperimentRunner:
    """Coordinates and executes standardized research experiments."""

    def __init__(self, output_dir: str = "artifacts/phase05_experiments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark_suite(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        schema: TableSchema,
        model_names: List[str],
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        num_samples: int = 500,
        experiment_id: str = "benchmark",
    ) -> Dict[str, Any]:
        """Runs multi-model benchmark across specified synthesizer names."""
        kwargs_map = model_kwargs or {}
        synthesizers = {}
        for name in model_names:
            kw = kwargs_map.get(name, {})
            synthesizers[name] = get_synthesizer(name, **kw)

        runner = BenchmarkRunner(schema=schema, num_samples=num_samples)
        results = runner.run(synthesizers=synthesizers, df_train=df_train, df_test=df_test)

        # Save structured results
        out_path = os.path.join(self.output_dir, f"{experiment_id}_results.json")
        with open(out_path, "w") as f:
            # Filter non-serializable elements
            serializable = {
                "comparison_table": results["comparison_table"].to_dict(orient="records"),
                "comparison_markdown": results["comparison_markdown"],
                "detailed": {
                    k: {
                        "fit_time_sec": v["fit_time_sec"],
                        "sample_time_sec": v["sample_time_sec"],
                        "throughput_rows_per_sec": v["throughput_rows_per_sec"],
                        "parameter_count": v["parameter_count"],
                        "fidelity_summary": v.get("fidelity", {}).get("summary", {}),
                        "utility_f1_parity": v.get("utility", {}).get("mean_f1_parity", 0.0),
                        "privacy_overall": v.get("privacy", {}).get("overall_privacy_risk", "UNKNOWN"),
                        "privacy_dcr": v.get("privacy", {}).get("dcr_and_nndr", {}),
                    }
                    for k, v in results["detailed"].items()
                },
            }
            json.dump(serializable, f, indent=2)

        return results
