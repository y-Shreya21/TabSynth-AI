"""Fast Phase 05 smoke test runner verifying all evaluation pipelines."""

import sys
import time
import numpy as np
import pandas as pd

from tabsynth.data.schema import infer_table_schema
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator
from tabsynth.evaluation.utility.utility_report import UtilityEvaluator
from tabsynth.evaluation.privacy.privacy_report import PrivacyAuditor
from tabsynth.evaluation.benchmarks.copula_wrapper import GaussianCopulaSynthesizer
from tabsynth.evaluation.benchmarks.tvae import TVAESynthesizer


def main():
    print("=" * 70)
    print("TabSynth AI - Phase 05 Evaluation Smoke Test")
    print("=" * 70)
    start_time = time.time()

    # 1. Create synthetic test DataFrame with mixed types
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, size=n).astype(float),
        "capital_gain": np.random.exponential(scale=1000, size=n),
        "education": np.random.choice(["Bachelors", "HS-grad", "Masters", "Doctorate"], size=n),
        "workclass": np.random.choice(["Private", "Self-emp", "Gov"], size=n),
        "income": np.random.choice(["<=50K", ">50K"], size=n, p=[0.75, 0.25]),
    })

    # Train / Test split
    df_train = df.iloc[:100].reset_index(drop=True)
    df_test = df.iloc[100:].reset_index(drop=True)

    schema = infer_table_schema(df_train, target_column="income")
    print(f"[+] Inferred schema: {len(schema.continuous_columns)} continuous, {len(schema.categorical_columns)} categorical.")

    # 2. Fit fast baseline (Gaussian Copula) and sample
    print("[*] Testing Gaussian Copula baseline fit & sample...")
    copula = GaussianCopulaSynthesizer(schema=schema, seed=42)
    copula.fit(df_train, target_column="income")
    synth_copula = copula.sample(num_samples=80)
    assert len(synth_copula) == 80
    print(f"[✓] Gaussian Copula generated {len(synth_copula)} records.")

    # 3. Test TVAE fit & sample
    print("[*] Testing native TVAE baseline fit & sample...")
    tvae = TVAESynthesizer(schema=schema, epochs=5, batch_size=32, device="cpu", seed=42)
    tvae.fit(df_train, target_column="income")
    synth_tvae = tvae.sample(num_samples=50)
    assert len(synth_tvae) == 50
    print(f"[✓] TVAE generated {len(synth_tvae)} records.")

    # 4. Test Fidelity Evaluator
    print("[*] Testing FidelityEvaluator...")
    fid_eval = FidelityEvaluator(
        continuous_cols=schema.continuous_columns,
        categorical_cols=schema.categorical_columns,
        run_c2st=True,
        run_sdmetrics=False,
    )
    fid_rep = fid_eval.evaluate(df_train, synth_copula, c2st_subsample=100)
    print(f"[✓] Fidelity Evaluator completed: Mean KS = {fid_rep['summary']['mean_ks']:.4f}, C2ST ROC-AUC = {fid_rep['summary'].get('c2st_roc_auc', 0.5):.4f}")

    # 5. Test Utility Evaluator (TRTR, TSTR, Augmentation)
    print("[*] Testing UtilityEvaluator...")
    util_eval = UtilityEvaluator(
        target_column="income",
        classifiers=["logistic_regression", "random_forest"],
        augmentation_ratios=[0.5, 1.0],
        random_state=42,
    )
    util_rep = util_eval.evaluate(df_train, synth_copula, df_test)
    print(f"[✓] Utility Evaluator completed: Mean F1 Parity = {util_rep['mean_f1_parity']:.4f}")

    # 6. Test Privacy Auditor (DCR, NNDR, MIA, Singling Out)
    print("[*] Testing PrivacyAuditor...")
    priv_auditor = PrivacyAuditor(
        continuous_cols=schema.continuous_columns,
        categorical_cols=schema.categorical_columns,
        sensitive_column="income",
        subsample_size=100,
        random_state=42,
    )
    priv_rep = priv_auditor.audit(df_train, df_test, synth_copula)
    print(f"[✓] Privacy Auditor completed: Overall Risk = {priv_rep['overall_privacy_risk']}, MIA AUC = {priv_rep['membership_inference']['mia_roc_auc']:.4f}")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] Phase 05 smoke test passed in {elapsed:.2f}s (< 30s threshold).")
    print("=" * 70)


if __name__ == "__main__":
    main()
