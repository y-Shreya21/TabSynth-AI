"""Predicate-based singling-out privacy risk evaluation."""

from typing import Dict, List, Any, Optional
import itertools
import numpy as np
import pandas as pd


def evaluate_singling_out(
    df_train: pd.DataFrame,
    df_synth: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    max_k_attributes: int = 3,
    max_predicates_per_k: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluates predicate-based singling-out risk under empirical privacy standards.
    
    A synthetic predicate isolates a single record in the synthetic data. The attack
    succeeds if that exact predicate also isolates exactly one real individual in
    the real training data.
    """
    rng = np.random.default_rng(random_state)
    
    # Pick quasi-identifiers: preferentially discrete/categorical features and low-cardinality continuous
    if quasi_identifiers is None:
        candidates = []
        for col in df_train.columns:
            if df_train[col].nunique() < 50:
                candidates.append(col)
        q_cols = candidates if len(candidates) >= 2 else list(df_train.columns[:4])
    else:
        q_cols = quasi_identifiers

    total_predicates_evaluated = 0
    singling_out_successes = 0

    results_by_order: Dict[str, Any] = {}

    for k in range(1, min(max_k_attributes + 1, len(q_cols) + 1)):
        all_combos = list(itertools.combinations(q_cols, k))
        if len(all_combos) > max_predicates_per_k:
            sampled_indices = rng.choice(len(all_combos), size=max_predicates_per_k, replace=False)
            selected_combos = [all_combos[i] for i in sampled_indices]
        else:
            selected_combos = all_combos

        k_tested = 0
        k_successes = 0

        for cols in selected_combos:
            cols_list = list(cols)

            # Find unique combinations in synthetic data
            synth_counts = df_synth[cols_list].value_counts()
            # Unique synthetic predicates (count == 1)
            synth_uniques = synth_counts[synth_counts == 1].index

            if len(synth_uniques) == 0:
                continue

            # Limit to at most 10 unique predicates per column combination for speed
            if len(synth_uniques) > 10:
                synth_uniques = synth_uniques[:10]

            for pred_val in synth_uniques:
                k_tested += 1
                total_predicates_evaluated += 1

                # Construct query filter on real training data
                if k == 1:
                    val = pred_val[0] if isinstance(pred_val, tuple) else pred_val
                    match_mask = (df_train[cols_list[0]] == val)
                else:
                    conditions = [df_train[col] == val for col, val in zip(cols_list, pred_val)]
                    match_mask = np.logical_and.reduce(conditions)

                real_match_count = int(np.sum(match_mask))
                if real_match_count == 1:
                    # Successful singling out
                    k_successes += 1
                    singling_out_successes += 1

        order_rate = float(k_successes / max(1, k_tested)) if k_tested > 0 else 0.0
        results_by_order[f"order_{k}"] = {
            "predicates_tested": k_tested,
            "singling_out_successes": k_successes,
            "success_rate": order_rate,
        }

    overall_rate = float(singling_out_successes / max(1, total_predicates_evaluated)) if total_predicates_evaluated > 0 else 0.0

    risk_level = "LOW_RISK" if overall_rate <= 0.05 else ("MODERATE_RISK" if overall_rate <= 0.15 else "HIGH_RISK")

    return {
        "total_predicates_tested": total_predicates_evaluated,
        "total_singling_out_successes": singling_out_successes,
        "singling_out_risk_rate": overall_rate,
        "risk_level": risk_level,
        "by_attribute_order": results_by_order,
    }
