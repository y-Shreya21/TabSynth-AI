# TabSynth AI — Technical Interview Preparation Guide
## Deep-Dive Questions & Answers for ML Research, Architecture & Systems Interviews

**Subject**: Hybrid Transformer-MLP Diffusion for Tabular Data  
**Roles Targeted**: Staff/Senior ML Research Engineer, Principal AI Architect, MLOps / Systems Engineer  
**Version**: `v0.9.0-rc.1` / Release Candidate  

---

## Section 1: Architecture & Model Design

### Q1.1: Why combine a Transformer with an MLP denoiser rather than using a pure Transformer or pure MLP?
**Answer**:
Tabular data lacks the spatial locality of images or the sequential ordering of text. Features are heterogeneous (different scales, types, semantics). 
- A **pure MLP** treats the table as a flat concatenated vector. While fast, it forces the network to learn feature identities and pairwise interactions implicitly through dense matrix multiplications, struggling to model high-order correlations across arbitrary column subsets (as shown in Kotelnikov et al., 2023). In our benchmarks, the flat MLP yielded a Pearson Frobenius error of **0.1591**.
- A **pure Transformer** operates on $D$ tokens with multi-head self-attention ($\mathcal{O}(D^2)$ complexity) and feed-forward networks on each token independently. However, tabular denoising requires synthesizing a joint tuple simultaneously; decoupling the tokens at the output layer makes it difficult to enforce coordinated multi-feature constraints during early reverse diffusion steps.
- **Our Hybrid Architecture** leverages the best of both:
  1. The **Pre-LN Transformer encoder** models cross-feature correlations by allowing every column token to attend to all other column tokens across the table.
  2. The contextualized tokens are concatenated and passed through a **deep Residual MLP backbone** that injects diffusion time and class conditioning vectors and outputs a unified joint prediction across continuous noise ($\boldsymbol{\epsilon}_\theta$) and categorical logits ($\hat{\mathbf{p}}_0$).
  This reduced Pearson correlation Frobenius error by **29.8%** (0.1117 vs 0.1591) compared to the pure MLP baseline.

### Q1.2: Why use Pre-LayerNorm (Pre-LN) instead of Post-LayerNorm (Post-LN)?
**Answer**:
In Post-LN Transformers ($x_{l} = \text{LayerNorm}(x_{l-1} + \text{SubLayer}(x_{l-1}))$), gradients passing through residual connections are scaled by layer norms, causing the gradient norm to vanish or explode in early layers unless a delicate learning rate warmup is applied (Xiong et al., ICML 2020). 
In Pre-LN Transformers ($x_l = x_{l-1} + \text{SubLayer}(\text{LayerNorm}(x_{l-1}))$), the identity shortcut preserves clean gradient backpropagation directly from output to input:
$$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=1}^L \frac{\partial \text{SubLayer}(\dots)}{\partial x_{l-1}}$$
This allows stable, unproblematic training from step 0 across variable diffusion timesteps $t \sim \text{Uniform}(1, T)$ without gradient explosion.

### Q1.3: How does feature-wise tokenization work for mixed numerical and categorical features?
**Answer**:
Each tabular column $j \in \{1, \dots, D\}$ is transformed into a $d_{\text{model}}$-dimensional vector representation:
1. **Continuous Features ($j \in \mathcal{C}$)**: A scalar float $x_j \in [-1, 1]$ is passed through a column-specific linear projection:
   $$\mathbf{e}_j = x_j \cdot \mathbf{W}_j + \mathbf{b}_j, \quad \mathbf{W}_j \in \mathbb{R}^{1 \times d_{\text{model}}}, \quad \mathbf{b}_j \in \mathbb{R}^{d_{\text{model}}}$$
2. **Categorical Features ($j \in \mathcal{D}$)**: An integer index $k \in \{0, \dots, K_j-1\}$ indexes a column-specific embedding matrix:
   $$\mathbf{e}_j = \mathbf{E}_j[k], \quad \mathbf{E}_j \in \mathbb{R}^{K_j \times d_{\text{model}}}$$
3. **Column Identity Embeddings**: To ensure the Transformer knows which feature is which regardless of position, a learnable column identifier $\mathbf{c}_j \in \mathbb{R}^{d_{\text{model}}}$ is added:
   $$\mathbf{z}_j^{(0)} = \mathbf{e}_j + \mathbf{c}_j$$
This forms a sequence of $D$ tokens $\mathbf{Z}^{(0)} \in \mathbb{R}^{B \times D \times d_{\text{model}}}$ ready for multi-head self-attention.

---

## Section 2: Diffusion Mathematics & Formulations

### Q2.1: Explain the continuous Gaussian diffusion forward and reverse processes.
**Answer**:
- **Forward Process**: Perturbs continuous features $\mathbf{x}_0$ into standard Gaussian noise over $T$ discrete steps according to a variance schedule $\beta_1, \dots, \beta_T$:
  $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$
  Using $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, any step $t$ can be sampled in closed form:
  $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- **Reverse Process**: Inverts the noise corruption by parameterizing a neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)$ to predict the injected noise $\boldsymbol{\epsilon}$:
  $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, c) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, c), \tilde{\beta}_t \mathbf{I}\right)$$
  $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) \right), \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$
  For $t > 1$, we sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and compute $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta + \sqrt{\tilde{\beta}_t} \mathbf{z}$. For $t = 1$, we set $\mathbf{z} = \mathbf{0}$.

### Q2.2: How does D3PM discrete multinomial diffusion work and why is it superior to one-hot Gaussian relaxation?
**Answer**:
Continuous diffusion on one-hot vectors treats categorical features as continuous points in $\mathbb{R}^K$. During reverse steps, intermediate states wander outside the probability simplex $\Delta^{K-1}$, generating negative values and non-normalized sums that require artificial post-hoc argmax or softmax rounding, introducing distortion.

**D3PM (Austin et al., 2021)** defines a true discrete Markov transition matrix $\mathbf{Q}_t \in \mathbb{R}^{K \times K}$ over discrete states $\{1, \dots, K\}$:
$$[\mathbf{Q}_t]_{mn} = q(x_t = n \mid x_{t-1} = m) = (1 - \beta_t)\delta_{mn} + \frac{\beta_t}{K}$$
The cumulative transition is $\bar{\mathbf{Q}}_t = \prod_{s=1}^t \mathbf{Q}_s$, and the forward marginal is:
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathbf{x}_0 \bar{\mathbf{Q}}_t$$
The exact analytical reverse transition Bayes posterior is:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{\mathbf{x}_t \mathbf{Q}_t \mathbf{x}_{t-1}^\top \cdot \mathbf{x}_0 \bar{\mathbf{Q}}_{t-1} \mathbf{x}_{t-1}^\top}{\mathbf{x}_0 \bar{\mathbf{Q}}_t \mathbf{x}_t^\top}$$
The model predicts the clean categorical distribution $\hat{\mathbf{p}}_0 = \text{Softmax}(\mathbf{z}_\theta(\mathbf{x}_t, t, c))$, and we marginalize over $\hat{\mathbf{p}}_0$:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \sum_{k=1}^K \hat{p}_{0, k} \cdot q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0 = \mathbf{e}_k)$$
Reverse transitions are always valid probability distributions on the simplex.

### Q2.3: How do linear, cosine, and cosine-log noise schedules differ?
**Answer**:
- **Linear Schedule (Ho et al., 2020)**: $\beta_t$ interpolates linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. While effective for images, on tabular data it destroys delicate continuous feature distributions too rapidly in the first 20% of steps.
- **Cosine Schedule (Nichol & Dhariwal, 2021)**: Sets $\bar{\alpha}_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$. It features a gentle slope at both small $t$ (preserving fine-grained structure) and large $t$ (preventing abrupt complete information destruction).
- **Cosine-Log Schedule**: Introduces a logarithmic dampening factor $s_{\text{log}}$ that slows down corruption of high-cardinality discrete categories, preventing premature convergence to uniform random noise.

---

## Section 3: Classifier-Free Guidance (CFG)

### Q3.1: What is Classifier-Free Guidance and how is it implemented in TabSynth AI?
**Answer**:
In standard Classifier Guidance (Dhariwal & Nichol, 2021), a separate time-dependent classifier $p(c \mid \mathbf{x}_t, t)$ is trained on noisy data, and its gradient $\nabla_{\mathbf{x}_t} \log p(c \mid \mathbf{x}_t)$ is added to the denoiser score during sampling. This requires training and tuning a separate neural network for every dataset.

**Classifier-Free Guidance (Ho & Salimans, 2021)** eliminates the separate classifier:
1. **Training Phase**: The denoiser is conditioned on class label $c$. With probability $p_{\text{uncond}} = 0.15$, $c$ is replaced by an unconditional mask token $\emptyset$. A single network learns both conditional $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)$ and unconditional $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)$ representations.
2. **Inference Phase**: We extrapolate the score using guidance scale $w \ge 0$:
   $$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + (1 + w) \cdot \left[ \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) \right]$$
   Setting $w = 0$ yields standard conditional sampling; setting $w > 1.0$ (e.g., $w = 1.5$) steers the sampling trajectory aggressively toward the specified class mode, greatly improving minority class sample quality.

### Q3.2: Why did balanced sampling with CFG produce a +48.55% minority recall gain in Ablation E5?
**Answer**:
Under natural conditioning on Adult Income (78% $\le 50K$, 22% $> 50K$), random sampling reflects the natural prior. Downstream Random Forest classifiers trained on such data minimize empirical risk by defaulting to majority class predictions, resulting in a minority recall of **0.0000**.
With balanced sampling ($50\%$ class $0$, $50\%$ class $1$) and CFG scale $w = 1.5$, reverse trajectories are explicitly guided to generate equal numbers of crisp minority instances with distinctive feature combinations (high capital gains, older age, advanced education). When the downstream Random Forest is trained on this balanced synthetic data, its minority recall rises from **0.0000 to 0.4855**—a direct **+48.55% gain**.

---

## Section 4: Statistical Fidelity & Metrics

### Q4.1: Why evaluate both 1D marginals and 2D/multi-D metrics?
**Answer**:
- **1D Marginals** (Kolmogorov-Smirnov for continuous, Total Variation Distance for categorical) verify that each individual column has the correct shape, mean, variance, and category proportions. However, a model that independently shuffles each column preserves 1D marginals perfectly while destroying all correlations.
- **2D Correlations** (Pearson Frobenius distance for linear associations, Spearman for rank monotonic associations, Theil's U / Cramer's V for categorical associations) verify that pairwise feature relationships are preserved.
- **Multivariate Distinguishability** (Classifier Two-Sample Test / C2ST) trains a binary classifier to discriminate real from synthetic rows. If the synthetic data matches the joint distribution $P(\mathbf{X})$, C2ST accuracy and ROC-AUC should be near $0.50$ (random guessing).

### Q4.2: Why do C2ST scores often reach near 1.0 on large tabular datasets, and does that mean the model failed?
**Answer**:
No, it does not mean the model failed.
As established in recent literature (Kotelnikov et al., 2023; Zhang et al., 2024), C2ST on tabular data with $N > 10,000$ and $D > 10$ features is hypersensitive. A strong classifier (e.g., LightGBM or Random Forest with 500 trees) can easily detect tiny distribution discrepancies in the 3rd or 4th decimal place of category probabilities or high-order interaction terms.
Even if marginal KS distances are $\le 0.05$ and correlation Frobenius distances are $\le 0.11$, the classifier will find a separating hyper-plane with $> 99\%$ accuracy. This is why C2ST must always be interpreted alongside marginal distances, downstream utility (TSTR), and DCR privacy metrics rather than in isolation.

---

## Section 5: Empirical Privacy Auditing

### Q5.1: How is Distance to Closest Record (DCR) formulated and interpreted?
**Answer**:
Let $\mathbf{s} \in \mathcal{S}$ be a synthetic record, and let $\mathcal{D}$ be a real dataset partition. The distance to the closest record is:
$$\text{DCR}(\mathbf{s}, \mathcal{D}) = \min_{\mathbf{x} \in \mathcal{D}} d_{\text{Gower}}(\mathbf{s}, \mathbf{x})$$
where $d_{\text{Gower}}$ is the normalized Gower distance across continuous and categorical columns.
- We compute $\text{DCR}(\mathbf{s}, \mathcal{D}_{\text{train}})$ and $\text{DCR}(\mathbf{s}, \mathcal{D}_{\text{test}})$.
- **DCR Ratio**:
  $$\text{DCR Ratio} = \frac{\mathbb{E}_{\mathbf{s} \in \mathcal{S}}[\text{DCR}(\mathbf{s}, \mathcal{D}_{\text{train}})]}{\mathbb{E}_{\mathbf{s} \in \mathcal{S}}[\text{DCR}(\mathbf{s}, \mathcal{D}_{\text{test}})]}$$
- **Interpretation**: If the model memorized training records, synthetic samples would be much closer to $\mathcal{D}_{\text{train}}$ than to unseen $\mathcal{D}_{\text{test}}$, causing the DCR ratio to drop significantly below $1.0$ (e.g., $< 0.8$). In TabSynth Hybrid, the DCR ratio is **1.0155** and the 5th percentile ratio is **2.4997**, proving that synthetic records are, on average, slightly further from training data than unseen test data is, indicating no memorization.

### Q5.2: What is the crucial difference between empirical privacy metrics and Differential Privacy?
**Answer**:
- **Empirical Privacy Metrics** (DCR, NNDR, exact match rate, shadow MIA ROC-AUC) measure the geometric separation and empirical memorization on a *specific finite sample*. They indicate whether the model replicated training records or if a simple nearest-neighbor attack can infer membership.
- **Differential Privacy (DP)** is a mathematical, worst-case guarantee:
  $$\mathbb{P}[\mathcal{M}(\mathcal{D}) \in \mathcal{S}] \le e^\epsilon \cdot \mathbb{P}[\mathcal{M}(\mathcal{D}') \in \mathcal{S}] + \delta$$
  for any two neighboring datasets $\mathcal{D}, \mathcal{D}'$ differing by a single record. DP protects against an adversary with *arbitrary auxiliary information and infinite compute*.
Empirical privacy is a valuable diagnostic, but it does **NOT** guarantee provable $(\epsilon, \delta)$-DP. Provable DP requires algorithmic noise addition (e.g., DP-SGD) during training.

---

## Section 6: Systems, MLOps & Production Engineering

### Q6.1: How does TabSynth AI handle high-concurrency database access without locking?
**Answer**:
TabSynth AI utilizes SQLite with enterprise-grade PRAGMA tuning in `api/db/session.py`:
- `PRAGMA journal_mode=WAL;`: Write-Ahead Logging allows readers and writers to operate concurrently. Readers read from the database file while writers append to the `-wal` log, eliminating reader-writer lock contention.
- `PRAGMA synchronous=NORMAL;`: Reduces fsync calls without risking database corruption in WAL mode.
- `PRAGMA busy_timeout=10000;`: If a transient write lock occurs, SQLite will wait up to 10 seconds before throwing an `OperationalError`.
- `PRAGMA foreign_keys=ON;`: Enforces relational integrity across jobs, models, and artifacts.
- `scripts/backup_db.py`: Uses the native `sqlite3.backup()` C-API to take consistent online hot backups during live traffic without locking read transactions.

### Q6.2: How are security risks (path traversal, malicious uploads) mitigated?
**Answer**:
1. **Path Traversal Confinement (`api/utils/security.py`)**:
   - Canonicalizes user-supplied paths via `os.path.realpath(os.path.abspath(path))`.
   - Validates that the canonical path starts with one of the registered storage roots (`settings.get_allowed_storage_roots()`).
   - Rejects traversal attempts (`../../etc/passwd`, null bytes `\x00`) with `HTTP 400 Bad Request`.
2. **Upload Sanitization**:
   - Enforces `MAX_UPLOAD_SIZE_BYTES = 100 MB`.
   - Strips dangerous characters from filenames via regex slugification.
   - Sniffs the first 512 bytes for null bytes or binary executable headers to reject disguised binaries.
3. **Pydantic Validation**:
   - Enforces `Field(gt=0)` on epochs, batch sizes, timesteps, and sample counts, preventing negative index bugs or zero-division crashes.
