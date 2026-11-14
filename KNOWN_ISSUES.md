# TabSynth AI: Known Issues & Technical Constraints

## 1. Storage Constraints
- **Issue:** Development machine has approximately 3.0 to 5.0 GB available storage.
- **Impact:** Heavy checkpointing or massive multi-gigabyte synthetic generations could exhaust disk space.
- **Mitigation:**
  - Enforce top-k rolling checkpointing (`keep_top_k: 2`).
  - Streamline virtual environment size and purge pip cache if needed (`pip cache purge` frees ~3.5 GB).
  - Clean intermediate experiment files promptly.

## 2. Apple Silicon MPS Precision
- **Issue:** `torch.float16` can experience numerical underflow or produce NaN during diffusion cumulative alpha multiplication ($\prod \alpha_s$) and multinomial matrix exponentiation.
- **Impact:** Gradient explosions or NaN predictions during denoising score matching.
- **Mitigation:** Enforce `torch.float32` across all core diffusion kernels, embeddings, and transition matrix calculations.

## 3. Sandboxed Network Isolation
- **Issue:** Standard sandboxed commands block outbound network connections to `pypi.org` and dataset archives.
- **Impact:** Package installation and UCI dataset downloading require manual approval with `BypassSandbox: true`.
- **Mitigation:** Execute only download and pip install commands with `BypassSandbox: true`; keep all code compilation, model training, and testing inside standard sandboxed execution.

## 4. Categorical Marginalization Scaling on High Cardinalities
- **Issue:** The exact probability marginalization $p_\theta(x_{t-1} \mid x_t) = \sum_k q(x_{t-1} \mid x_t, x_0 = e_k) p_\theta(x_0 = e_k \mid x_t)$ evaluates across all $K$ basis states, scaling with computational complexity $\mathcal{O}(B \cdot K^2)$ per column.
- **Impact:** For columns with extreme cardinalities ($K > 500$), memory and compute consumption can become significant.
- **Mitigation:** For standard tabular datasets (e.g. Adult Income, Bank Marketing with $K \le 50$), this runs in milliseconds. For extreme cardinality datasets, preprocessor frequency truncation or clustering should be applied.

## 5. Noise Schedule Selection for Low Timestep Regimes ($T \le 100$)
- **Issue:** The standard linear beta schedule ($\beta_{\text{start}}=10^{-4}, \beta_{\text{end}}=0.02$) was originally parameterized for $T=1000$. When executed at $T=100$, $\bar{\alpha}_{99} \approx 0.36$, retaining non-trivial signal at the final step.
- **Impact:** Incomplete corruption at $t=T-1$ under linear schedule when $T$ is small.
- **Mitigation:** Default to the Cosine beta schedule (`schedule_type: "cosine"`), which analytically scales with $T$ and guarantees $\bar{\alpha}_{T-1} \approx 10^{-7} \approx 0$ regardless of $T$.

## 6. Apple Silicon MPS Unified Memory Caching
- **Issue:** Repeated batch allocations and backward passes on Apple Silicon MPS can cause gradual memory pool fragmentation in PyTorch Metal backend.
- **Impact:** Virtual memory pressure during long training runs or large-batch sampling jobs.
- **Mitigation:** `TabSynthTrainer` and `GenerationPipeline` explicitly trigger `torch.mps.empty_cache()` at epoch boundaries and following large generation chunks when running on MPS.

