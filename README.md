# MI-CNN — Mamba-Inspired CNN for Postural State Classification in Immersive VR

Reference implementation of **MI-CNN**, the model proposed in:

> **Toward Postural State Classification in Immersive VR with Multimodal Data and Explainability Analysis**
> Nipa Anjum, Md Irfan Pavel, Robert Gonzalez Jr, Kevin Desai, Alberto Cordova, M. Rasel Mahmud, John Quarles
> *IEEE International Symposium on Mixed and Augmented Reality (ISMAR), Bari, Italy, 2026*

MI-CNN classifies short windows of multimodal VR sensor data (full-body kinematics, EMG, and electrodermal activity) as either a **balanced** or an **imbalanced** postural state. In the paper it reaches **0.968 ± 0.015 accuracy** and **0.996 ± 0.003 ROC-AUC** under Leave-One-Participant-Out cross-validation, the best of the eight machine learning and deep learning models compared.

This repository contains the **model and its evaluation protocols only** — the novel contribution of the work. The comparative baselines (Logistic Regression, Random Forest, XGBoost, LightGBM, Extra Trees, CNN-LSTM, TCN), the SHAP explainability analysis, and all result tables and figures are described in the paper and are not reproduced here.

---

## The model

MI-CNN keeps the **split–gate–project block structure of Mamba** ([Gu & Dao, 2024](https://arxiv.org/abs/2312.00752)) but replaces the selective state-space recurrence with a **depthwise causal convolution and a learned gate**, in the spirit of Gated Convolutional Networks ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)).

The result is a lightweight, fully parallel TensorFlow model: no recurrent scan, no custom CUDA kernel, and no `mamba-ssm` dependency — it runs anywhere TensorFlow runs.

```
input window (T=100, F=1001)
    │
    ├─ Dense(64) + LayerNorm                       input projection
    │
    ├─ 4 × MambaInspiredBlock(d_model=64)          sequence encoder
    │      │
    │      ├─ Dense(256, no bias) ──► split ──► x (128) , gate branch (128)
    │      │                                    │
    │      │        x:  SiLU → depthwise causal Conv1D(k=3) → SiLU
    │      │            × sigmoid(softplus(Dense(128)))     ← selective gate
    │      │            × SiLU(gate branch)                 ← GLU interaction
    │      │
    │      └─ Dense(64, no bias) → Dropout(0.3) → + residual → LayerNorm
    │
    ├─ GlobalAveragePooling1D                      (B, 64)
    ├─ Dense(128, relu) → Dropout(0.3)
    ├─ Dense(64,  relu) → Dropout(0.3)
    └─ Dense(2, softmax)
```

**Why these choices.** The convolution is *depthwise* so each of the 128 inner channels is filtered independently, keeping the block cheap across a very wide input. It is *causal* so position `t` never sees the future, which leaves the architecture usable for streaming inference in a live VR system. The gate is what makes the block selective: `sigmoid(softplus(·))` produces a per-channel, per-timestep weight that decides how much of the convolved signal to retain, which is the role the state-space recurrence plays in Mamba.

**Size.** With `F = 1001` input features the model has **256,066 trainable parameters** — about 1.0 MB of weights, or the **3.07 MB** reported in the paper when saved as a trained `.keras` checkpoint that also carries the Adam optimizer state. Reported single-window inference: **3.72 ± 0.46 ms** on GPU and **9.45 ± 1.21 ms** on CPU — under the windowing used here, a live system would produce an updated prediction every ~0.83 s.

> **Note on `x_proj`.** Each block instantiates a state-projection layer `x_proj` that is computed but does not feed the block output; the selective behaviour is carried entirely by the `dt_proj` gate. It is retained here because it is part of the trained model and of the 256,066-parameter count reported in the paper. TensorFlow will print a `Gradients do not exist for variables ['.../x_proj/kernel']` warning during training — this is expected and harmless, not a misconfiguration.

---

## Repository layout

```
micnn/
  model.py        MambaInspiredBlock and build_micnn
  data.py         loading, windowing, participant-wise balancing, scaling
  evaluation.py   Leave-One-Participant-Out and Leave-One-Scenario-Out protocols
  scenarios.py    35 perturbation labels → 7 fall categories
  config.py       all hyperparameters used for the reported results
scripts/
  run_lopo.py         subject-independent evaluation (headline result)
  run_loso.py         scenario-independent evaluation (robustness analysis)
  measure_latency.py  parameter count, model size, single-window latency
examples/
  quickstart.py   end-to-end smoke test on synthetic data — no dataset needed
requirements.txt  dependency version floors
pyproject.toml    package metadata, for `pip install -e .`
LICENSE           MIT
```

## Installation

```bash
git clone https://github.com/NipaAnjum/MI-CNN.git
cd MI-CNN
pip install -r requirements.txt
```

The scripts in `scripts/` and `examples/` run directly from a clone, from any working directory. To use `micnn` as a library from elsewhere, install it into your environment:

```bash
pip install -e .
```

Requires Python 3.9+ and TensorFlow 2.12+. A GPU is recommended but not required; the code falls back to CPU automatically.

Verify the installation without any data — this builds the model, checks the parameter count, and runs both evaluation protocols on a small synthetic recording:

```bash
python examples/quickstart.py
```

## Data

The experiments use the **public multimodal VR balance dataset** of Ribeiro et al., *Immersive Ecological Virtual Environment for Inducing Balance Disturbances* (IEEE TVCG, 2025). It is **not redistributed here** — please obtain it from the original authors.

The code expects one wide CSV where each row is a synchronised 60 Hz sample:

| Column | Meaning |
| --- | --- |
| `Participant` | participant identifier, e.g. `Participant1` |
| `Timestamp_Xsens` | timestamp string, `HH:MM:SS.mmm` |
| `Frame` | Xsens frame counter |
| `Label` | perturbation id, `0`–`35` (`0` = no perturbation) |
| `class` | target — `0` balanced, `1` imbalanced |
| `Shimmer_Acc_y`, `Shimmer_Acc_z` | present in the CSV, but dropped by `load_dataset()` (see below) |
| *(all others)* | sensor channels: Xsens kinematics, EMG, Shimmer inertial/physiological |

Preprocessing to apply *before* this pipeline, following the paper: remove duplicate and inconsistent rows, and discard Participant 12 (its recordings were identical to Participant 11, which would have leaked data across folds). That leaves **11 participants and 1003 sensor columns**.

The pipeline itself then drops the two Shimmer accelerometer axes (`Shimmer_Acc_y`, `Shimmer_Acc_z`) that are missing for Participant 3, so they are removed for everyone and the feature space stays identical across participants. This happens inside `load_dataset()` — see `DROP_COLUMNS` in [`micnn/config.py`](micnn/config.py) — and yields the **1001 feature columns** reported in the paper (982 Xsens + 12 Shimmer + 7 EMG). Leave those two columns in your CSV; the code removes them.

If your CSV is not yet class-balanced, pass `--downsample` to `scripts/run_lopo.py` to apply the participant-wise balancing described below. (`run_loso.py` expects an already-balanced CSV.)

## Usage

**Leave-One-Participant-Out** — the headline protocol. Each fold holds out one participant entirely, so every score is measured on a person the model has never seen:

```bash
python scripts/run_lopo.py --data final_dataset.csv --output results/micnn-lopo.csv
```

Reproduce the non-overlapping-window robustness check by setting the stride equal to the window length:

```bash
python scripts/run_lopo.py --data final_dataset.csv --step-size 100
```

**Leave-One-Scenario-Out** — each fold removes one whole perturbation category from training and tests detection of that never-seen kind of instability:

```bash
python scripts/run_loso.py --data final_dataset.csv --output-dir results --plot
```

**Latency and model size:**

```bash
python scripts/measure_latency.py --features 1001 --runs 1000
```

**As a library:**

```python
from micnn import Config, load_dataset, create_windows, run_lopo, summarize

config = Config()
df = load_dataset("final_dataset.csv")
X, y, participants, features = create_windows(df, config.window_size, config.step_size)

results = run_lopo(X, y, participants, config)
print(summarize(results))
```

---

## Method details

**Windowing.** Fixed-length sliding windows of **100 samples** (~1.67 s at 60 Hz) with a stride of **50** (~0.83 s, 50% overlap). Each window takes a single label by majority vote. Windows are generated **within each participant only**, so no window ever spans two people.

**Class balancing.** The raw recordings are roughly **5:1** in favour of the non-perturbed class overall, ranging from 3.3:1 to 8:1 across individual participants. Balancing is done **per participant** — the smaller class is kept in full and an equal number of samples is drawn from the larger one, which for this dataset means every class-1 sample is retained — so that each individual keeps their share of the dataset and inter-individual variability is not distorted by the resampling.

**Leakage controls.** These are the parts of the pipeline that matter most for trusting the numbers:

- The `StandardScaler` is fit on the **training fold only** and merely applied to the test fold. Fitting on pooled data would let test statistics inform training.
- Timestamp columns are **excluded from the feature set**. This matters more than it looks: because balancing happens before windowing (see below), the inter-sample time gap alone separates the two classes almost perfectly, so leaving any timestamp in the features would be a straight shortcut.
- Because windows are cut within participants and folds are split by participant, **overlapping windows can never straddle the train/test boundary** under LOPO.
- In the scenario protocol, any window that straddles a perturbation onset/offset or mixes two categories is **dropped**, so no fragment of the held-out scenario survives in training. Note this guarantee covers the held-out *scenario*: a control test window adjacent to a perturbation onset can still overlap a positive training window, but the shared rows are all `Label == 0`. That can slightly flatter `accuracy` and `control_recall` in the scenario protocol; it does not affect `scenario_recall`.
- Control windows in the scenario protocol are split **contiguously in time** per participant with a one-window guard gap, never randomly — a random split of 50%-overlapping windows would place two halves of the same signal on opposite sides of the boundary. The same control split is reused in every fold, so the only variable across folds is which scenario is held out.
- Each fold rebuilds and reseeds the model from scratch; no state carries over.

**Why Leave-One-Participant-Out rather than k-fold.** Under standard k-fold, windows from the same person appear in both training and test sets, which inflates accuracy and answers the wrong question. LOPO measures what actually matters for deployment: how the model behaves on a *new user*. The per-fold spread it produces is itself a result — it quantifies how much balance-control strategy varies between individuals.

**Why Leave-One-Scenario-Out as well.** Every participant experiences the same set of visual perturbations, so participant-wise validation alone cannot rule out the model recognising scenario-specific structure rather than a general signature of instability. Holding out an entire perturbation *category* tests exactly that. The headline metric is `scenario_recall`: the fraction of windows from the never-seen perturbation type still flagged as imbalanced. The seven categories are lateral (roll/ML), forward/backward (AP), slip (pitch), trip, fall from heights (vertigo), object avoidance, and syncope.

The scenario run also writes a **per-sub-scenario breakdown**, which reports the detection rate for each of the 35 original perturbations inside a held-out category. This is what explains *why* a category scores lower — without any retraining, since every one of those windows is a true positive.

---

## Reproducibility notes and known limitations

This code reproduces the published experiments exactly, which means it also reproduces their design choices. The following are properties of the protocol that a reader should know about; they are documented here rather than quietly changed, because changing them would no longer reproduce the paper.

**Class balancing happens before windowing, so control windows are not contiguous in time.** Downsampling removes individual class-0 *samples*, and windows are cut from what remains. A perturbed window really is 100 consecutive samples (~1.67 s, confirmed by measuring the released data). A control window is 100 consecutive *retained rows*, which on the real dataset spans a median of roughly 5–14 s of wall clock depending on the participant, with gaps. The "~1.67 s" figure therefore describes the perturbed class, not both.

We checked whether this is exploitable. The figures below come from a diagnostic run on this pipeline and are not results reported in the paper. The inter-sample time gap alone separates the classes at ~0.98 AUC, which is exactly why timestamps are excluded from the features. From the sensor channels themselves, the obvious proxies for the dilation are weak: mean absolute lag-1 difference per window gives 0.43 AUC and within-window variance 0.50, against the 0.968 accuracy the model achieves. So the artifact is unlikely to explain the results, though this is evidence rather than proof. Cutting windows before balancing would avoid the issue entirely and is the recommended design for follow-up work.

**Early stopping uses the tail of the training fold, not a stratified sample.** Keras's `validation_split` takes the last 10% of rows *before* shuffling, and the fold arrays are ordered rather than shuffled. Two consequences:

- Under LOPO, the validation set is the late portion of the last training participant's session. It contains both classes, but it is one person rather than a sample of the fold.
- Under leave-one-scenario-out, training rows are ordered positives-then-controls, so the validation set is **entirely control windows**. Early stopping and LR decay are therefore driven by negative-class loss alone, which selects against positive predictions and biases `scenario_recall` **downward**. The reported scenario numbers are conservative in this respect, not optimistic.

Passing an explicit stratified `validation_data=` to `_fit_fold` would fix both; it is deliberately not done here so the defaults match the published runs.

**Windows can span trial boundaries.** Windowing is continuous within a participant, so a window may bridge two recording segments where the timestamp jumps. This affects both classes.

**Scope.** Eleven healthy young adults under controlled visual perturbations. Nothing here is validated for older adults, clinical populations, or genuine physical slips and trips, and the evaluation is offline — the latency benchmark measures the model, not an end-to-end streaming system.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{anjum2026postural,
  title        = {Toward Postural State Classification in Immersive VR with
                  Multimodal Data and Explainability Analysis},
  author       = {Anjum, Nipa and Pavel, Md Irfan and Gonzalez Jr, Robert and
                  Desai, Kevin and Cordova, Alberto and Mahmud, M. Rasel and
                  Quarles, John},
  booktitle    = {2026 IEEE International Symposium on Mixed and Augmented
                  Reality (ISMAR)},
  year         = {2026},
  address      = {Bari, Italy},
  organization = {IEEE}
}
```

Please also cite the dataset authors if you use the data:

```bibtex
@article{ribeiro2025immersive,
  title   = {Immersive Ecological Virtual Environment for Inducing Balance Disturbances},
  author  = {Ribeiro, N. F. and Veloso, A. and Pires, H. and Santos, C. P.},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2025}
}
```

## Acknowledgments

This work was supported by a grant from the National Science Foundation (IIS 2403411).

## License

Released under the [MIT License](LICENSE). The dataset is licensed separately by its original authors.
