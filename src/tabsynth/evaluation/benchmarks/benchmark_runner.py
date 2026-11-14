"""Benchmark runner evaluating multiple synthesizers across fidelity, utility, and privacy."""

import time
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

from tabsynth.data.schema import TableSchema
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator
from tabsynth.evaluation.utility.utility_report import UtilityEvaluator
from tabsynth.evaluation.privacy.privacy_report import PrivacyAuditor
from .base import BaseSynthesizer
from .baseline_registry import get_synthesizer


class BenchmarkRunner:
    """Orchestrates comprehensive multi-model benchmarking."""

    def __init__(
        self,
        schema: TableSchema,
        num_samples: int = 500,
        run_fidelity: bool = True,
        run_utility: bool = True,
        run_privacy: bool = True,
        subsample_size: int = 1000,
        random_state: int = 42,
    ):
        self.schema = schema
        self.num_samples = num_samples
        self.run_fidelity = run_fidelity
        self.run_utility = run_utility
        self.run_privacy = run_privacy
        self.subsample_size = subsample_size
        self.random_state = random_state

        self.fidelity_evaluator = FidelityEvaluator(
            continuous_cols=schema.continuous_columns,
            categorical_cols=schema.categorical_columns,
            run_c2st=True,
            run_sdmetrics=False,
        )

        self.utility_evaluator = (
            UtilityEvaluator(
                target_column=schema.target_column,
                classifiers=["logistic_regression", "random_forest", "hist_gradient_boosting"],
                random_state=random_state,
            )
            if schema.target_column
            else None
        )

        self.privacy_auditor = PrivacyAuditor(
            continuous_cols=schema.continuous_columns,
            categorical_cols=schema.categorical_columns,
            sensitive_column=schema.target_column,
            subsample_size=subsample_size,
            random_state=random_state,
        )

    def run(
        self,
        synthesizers: Dict[str, BaseSynthesizer],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Executes full benchmark evaluation across all provided synthesizers."""
        results: Dict[str, Any] = {}
        comparison_records: List[Dict[str, Any]] = []

        for name, synth in synthesizers.items():
            print(f"\n[+] Running Benchmark for: {name}")

            # 1. Fit Synthesizer if needed
            fit_time = synth.training_time_sec
            if not synth.is_fitted:
                t0 = time.time()
                synth.fit(df_train, target_column=self.schema.target_column)
                fit_time = time.time() - t0

            # 2. Sample Generation & Throughput
            t_sample_start = time.time()
            df_synth = synth.sample(num_samples=self.num_samples)
            sample_time = time.time() - t_sample_start
            throughput = len(df_synth) / max(1e-6, sample_time)

            model_results: Dict[str, Any] = {
                "name": name,
                "fit_time_sec": float(fit_time),
                "sample_time_sec": float(sample_time),
                "throughput_rows_per_sec": float(throughput),
                "parameter_count": int(synth.parameter_count),
            }

            # 3. Fidelity Evaluation
            if self.run_fidelity:
                f_rep = self.fidelity_evaluator.evaluate(df_train, df_synth)
                model_results["fidelity"] = f_rep
            else:
                f_rep = {}

            # 4. Utility Evaluation
            if self.run_utility and self.utility_evaluator is not None and self.schema.target_column:
                u_rep = self.utility_evaluator.evaluate(df_train, df_synth, df_test)
                model_results["utility"] = u_rep
            else:
                u_rep = {}

            # 5. Privacy Evaluation
            if self.run_privacy:
                p_rep = self.privacy_auditor.audit(df_train, df_test, df_synth)
                model_results["privacy"] = p_rep
            else:
                p_rep = {}

            results[name] = model_results

            # Build row for comparison table
            f_sum = f_rep.get("summary", {})
            u_tstr = u_rep.get("tstr", {})
            p_dcr = p_rep.get("dcr_and_nndr", {})
            p_mia = p_rep.get("membership_inference", {})

            # RF TSTR metrics
            rf_res = u_tstr.get("random_forest", {})
            rf_f1_parity = rf_res.get("relative_f1_parity", 0.0)
            rf_tstr_f1 = rf_res.get("tstr", {}).get("macro_f1", 0.0)

            comparison_records.append({
                "Model": name,
                "Params": synth.parameter_count,
                "Throughput (rows/s)": round(throughput, 1),
                "Mean KS ↓": round(f_sum.get("mean_ks", 1.0), 4),
                "Mean Wasserstein ↓": round(f_sum.get("mean_wasserstein", 1.0), 4),
                "Mean TVD ↓": round(f_sum.get("mean_tvd", 1.0), 4),
                "Correlation Frob. ↓": round(f_sum.get("pearson_frobenius_distance", 1.0), 4),
                "C2ST ROC-AUC (~0.5)": round(f_sum.get("c2st_roc_auc", 1.0), 4),
                "TSTR RF Macro-F1 ↑": round(rf_tstr_f1, 4),
                "TSTR/TRTR F1 Parity ↑": round(rf_f1_parity, 4),
                "Mean DCR Train": round(p_dcr.get("mean_dcr_train", 0.0), 4),
                "DCR Ratio (~1.0)": round(p_dcr.get("dcr_ratio", 0.0), 4),
                "NNDR Mean": round(p_dcr.get("mean_nndr", 0.0), 4),
                "Exact Matches": p_dcr.get("exact_record_matches", 0),
                "MIA ROC-AUC (~0.5)": round(p_mia.get("mia_roc_auc", 0.5), 4),
                "Privacy Risk": p_rep.get("overall_privacy_risk", "UNKNOWN"),
            })

        df_comparison = pd.DataFrame(comparison_records)

        # Native markdown table generator without tabulate dependency
        headers = list(df_comparison.columns)
        md_lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df_comparison.iterrows():
            md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        md_table = "\n".join(md_lines)

        return {
            "detailed": results,
            "comparison_table": df_comparison,
            "comparison_markdown": md_table,
        }

