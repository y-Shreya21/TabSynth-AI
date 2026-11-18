"""Service coordinating statistical fidelity, downstream utility, and privacy auditing benchmarks."""

import os
import uuid
import logging
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import HTTPException, status
from api.config import settings
from api.db.models import ExperimentRecord
from api.db.repositories import ExperimentRepository
from api.utils.security import validate_safe_path
from tabsynth.data.schema import infer_table_schema
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy

logger = logging.getLogger("tabsynth.evaluation")


class EvaluationService:
    """Manages empirical evaluation benchmarks against synthetic tabular records."""

    @staticmethod
    def evaluate_fidelity(
        real_data_path: str,
        synthetic_data_path: str,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute marginal distributions, correlation distances, and C2ST indistinguishability."""
        allowed_roots = settings.get_allowed_storage_roots()
        safe_real = validate_safe_path(real_data_path, allowed_roots)
        safe_synth = validate_safe_path(synthetic_data_path, allowed_roots)

        if not os.path.exists(safe_real):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Real data path '{real_data_path}' was not found.",
            )
        if not os.path.exists(safe_synth):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Synthetic data path '{synthetic_data_path}' was not found.",
            )

        df_real = pd.read_csv(safe_real)
        df_synth = pd.read_csv(safe_synth)
        schema = infer_table_schema(df_real, target_column=target_column)

        results = evaluate_statistical_fidelity(
            real_df=df_real,
            synth_df=df_synth,
            continuous_cols=schema.continuous_columns,
            categorical_cols=schema.categorical_columns,
        )

        # Record experiment
        exp_id = f"exp_fid_{uuid.uuid4().hex[:8]}"
        ExperimentRepository.create(
            ExperimentRecord(
                id=exp_id,
                name=f"Fidelity_{schema.dataset_name}",
                description=f"Fidelity evaluation between {os.path.basename(real_data_path)} and {os.path.basename(synthetic_data_path)}",
                metrics=results,
            )
        )
        return results

    @staticmethod
    def evaluate_utility(
        real_train_path: str,
        synthetic_train_path: str,
        real_test_path: str,
        target_column: str,
    ) -> Dict[str, Any]:
        """Compute TSTR vs TRTR downstream classification metrics."""
        allowed_roots = settings.get_allowed_storage_roots()
        safe_train_real = validate_safe_path(real_train_path, allowed_roots)
        safe_train_synth = validate_safe_path(synthetic_train_path, allowed_roots)
        safe_test_real = validate_safe_path(real_test_path, allowed_roots)

        for p_orig, p_safe in [
            (real_train_path, safe_train_real),
            (synthetic_train_path, safe_train_synth),
            (real_test_path, safe_test_real),
        ]:
            if not os.path.exists(p_safe):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Required dataset file '{p_orig}' was not found.",
                )

        df_train_real = pd.read_csv(safe_train_real)
        df_train_synth = pd.read_csv(safe_train_synth)
        df_test_real = pd.read_csv(safe_test_real)

        results = evaluate_downstream_utility(
            df_train_real=df_train_real,
            df_train_synth=df_train_synth,
            df_test_real=df_test_real,
            target_column=target_column,
        )

        exp_id = f"exp_util_{uuid.uuid4().hex[:8]}"
        ExperimentRepository.create(
            ExperimentRecord(
                id=exp_id,
                name=f"Utility_TSTR_{target_column}",
                description="Downstream classification utility evaluation (TSTR vs TRTR)",
                metrics=results,
            )
        )
        return results

    @staticmethod
    def evaluate_privacy(
        real_train_path: str,
        real_test_path: str,
        synthetic_path: str,
        subsample_size: int = 1000,
    ) -> Dict[str, Any]:
        """Compute empirical Distance-to-Closest-Record, exact matches, and shadow MIA."""
        allowed_roots = settings.get_allowed_storage_roots()
        safe_train = validate_safe_path(real_train_path, allowed_roots)
        safe_test = validate_safe_path(real_test_path, allowed_roots)
        safe_synth = validate_safe_path(synthetic_path, allowed_roots)

        for p_orig, p_safe in [
            (real_train_path, safe_train),
            (real_test_path, safe_test),
            (synthetic_path, safe_synth),
        ]:
            if not os.path.exists(p_safe):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Required dataset file '{p_orig}' was not found.",
                )

        df_train = pd.read_csv(safe_train)
        df_test = pd.read_csv(safe_test)
        df_synth = pd.read_csv(safe_synth)
        schema = infer_table_schema(df_train)

        results = evaluate_empirical_privacy(
            df_train=df_train,
            df_test=df_test,
            df_synth=df_synth,
            continuous_cols=schema.continuous_columns,
            categorical_cols=schema.categorical_columns,
            subsample_size=subsample_size,
        )

        exp_id = f"exp_priv_{uuid.uuid4().hex[:8]}"
        ExperimentRepository.create(
            ExperimentRecord(
                id=exp_id,
                name="Privacy_Audit",
                description="Empirical Distance-to-Closest-Record and shadow MIA privacy audit",
                metrics=results,
            )
        )
        return results
