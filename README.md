# Conditional Fourier Diffusion

This repository contains the experiments used in our paper on sliding-window covariance estimation. This is a minimal public subset centered on `notebooks/paper.ipynb`; datasets and generated assets are not included.

## 1. Environment Setup

### 1.1 Conditional diffusion (cfdiff)
```bash
conda create -n cfdiff python=3.10 -y
conda activate cfdiff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install pot ipywidgets lightning[extra]
```

### 1.2 Baselines (PyPortfolioOpt + ARCH)
```bash
conda create -n baselines python=3.10 -y
conda activate baselines
pip install arch PyPortfolioOpt statsmodels
```
Use `conda run -n baselines ...` to execute baseline commands so the main environment is untouched.

Some scripts require additional libraries (e.g., `gluonts`, `pandas`, `matplotlib`, `Pillow`, `requests`, `PyYAML`). Install them as needed.

## 2. Data
Default data set is `exchange_rate_clean`. Set the data path before running experiments:
```bash
export CFDIFF_DATA_DIR=$HOME/.gluonts/datasets/exchange_rate_clean
```
Use `scripts/prepare_*_dataset.py` to build datasets. The iShares CSV inputs are not included; place them under `ishares/` before running `scripts/prepare_ishares_dataset.py`.

## 3. Training and Sampling
The notebook `notebooks/paper.ipynb` lists the commands used for the paper. Key examples:

**Conditional · time domain (train + sample)**
```bash
python -m cfdiff.cmd.train experiment=time ...
python -m cfdiff.cmd.sample experiment=time checkpoint_path=... output_dir=...
```

**Conditional · frequency domain**
```bash
python -m cfdiff.cmd.train experiment=fourier ...
python -m cfdiff.cmd.sample experiment=fourier ...
```

**Unconditional (fdiff)**
```bash
python -m fdiff.cmd.train experiment=unconditional ...
python -m fdiff.cmd.sample experiment=unconditional ...
```
Use the overrides listed in `paper.ipynb` to match the paper (context_len=60, pred_len=30, Beta t-sampling, etc.).

## 4. Baselines
To run the classical estimators (sample covariance, Ledoit–Wolf, RiskMetrics) across validation+test windows:
```bash
conda run -n baselines python src/baselines/cmd/run.py \
  +experiment=time \
  datamodule.data_dir=$CFDIFF_DATA_DIR \
  datamodule.context_len=60 datamodule.pred_len=30 \
  datamodule.stride=1 datamodule.val_ratio=0.3 datamodule.batch_size=32 \
  datamodule.num_workers=8 \
  baseline.method=sample_cov \
  baseline.include_val=true \
  baseline.window_index=-1 \
  baseline.save_csv=true baseline.save_pt=true
```
Replace `baseline.method` with `ledoit_wolf` or `riskmetrics` for the other baselines. The new `include_val=true` flag ensures the saved `winXXXX` files use the same global indices (0–47 validation, 48–159 test) as the diffusion runs.

Additional baselines from *A Deep Learning Framework for Medium-Term Covariance Forecasting* are wired into the same CLI:

- `baseline.method=ewma` for the plain exponential smoothing baseline (RiskMetrics-style, `lam` in `method_kwargs`).
- `baseline.method=ccc_garch` to fit per-asset GARCH(1,1) with a constant correlation matrix; the runner now defaults `method_kwargs.horizon` to `${datamodule.pred_len}` so forecasts align with the target window.
- `baseline.method=cab` to train the CNN–BiLSTM covariance model on the datamodule's training split and evaluate on validation+test with the same preprocessing. Example:
  ```bash
  conda run -n baselines python src/baselines/cmd/run.py \
    +experiment=time \
    datamodule.data_dir=$CFDIFF_DATA_DIR datamodule.context_len=60 datamodule.pred_len=30 \
    baseline.method=cab \
    baseline.method_kwargs.lookback_window=${datamodule.context_len} \
    baseline.method_kwargs.cov_window=8 \
    baseline.method_kwargs.epochs=5 \
    baseline.method_kwargs.device=cuda
  ```
  Keep `fourier_transform=false` so the CAB baseline sees time-domain windows; the runner handles standardisation, training split usage, and the usual metrics/CSV/PT dumps so outputs stay aligned with the diffusion experiments.

## 5. Plotting / Metrics
See `notebooks/paper.ipynb` for the exact commands. Common scripts include:

- `scripts/plot_stylized_facts.py`
- `scripts/plot_bull_bear_regimes.py`
- `scripts/plot_dataset_spectrum.py`
- `scripts/plot_eigen_windows.py`
- `scripts/plot_asset_panel.py`
- `scripts/plot_pair_window_metric.py`
- `scripts/plot_correlation_table.py`
- `scripts/make_dataset_summary.py`, `scripts/make_window_count_table.py`
- `scripts/make_paper_metric_tables.py`, `scripts/compute_loglik_regret.py`
- `scripts/compute_cross_signal_energy.py`

Each script has CLI help (`-h`).

## 6. Directory Layout
```
outputs/
  time/
    conditional/<timestamp>/...
    unconditional/<timestamp>/...
  fourier/
    conditional/<timestamp>/...
    unconditional/<timestamp>/...
outputs/baselines/<method>/<mode>/<timestamp>/...
assets/  # plotted figures
```

Set `CFDIFF_DATA_DIR` and use the commands above to reproduce the results.
