# Conditional Fourier Diffusion

This repository contains the experiments used in our paper on sliding-window covariance estimation. This is a minimal public subset; datasets and generated assets are not included. For step-by-step reproduction, start with `notebooks/reproduce_thesis.ipynb` and use `notebooks/paper.ipynb` as the full command log.

## 1. Environment Setup

### 1.1 Conditional diffusion (cfdiff)
```bash
conda create -n cfdiff python=3.10 -y
conda activate cfdiff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install pot ipywidgets lightning[extra] gluonts pandas matplotlib seaborn requests PyYAML tqdm pillow
```

### 1.2 Baselines (PyPortfolioOpt + ARCH)
```bash
conda create -n baselines python=3.10 -y
conda activate baselines
pip install arch PyPortfolioOpt statsmodels gluonts pandas matplotlib seaborn
```
Use `conda run -n baselines ...` to execute baseline commands so the main environment is untouched.

Some scripts require additional libraries (e.g., `gluonts`, `pandas`, `matplotlib`, `Pillow`, `requests`, `PyYAML`). Install them as needed.

## 2. Data
Default data set is `exchange_rate_clean`. Set the data path before running experiments:
```bash
export CFDIFF_DATA_DIR=$HOME/.gluonts/datasets/exchange_rate_clean
```
Use the dataset preparation scripts to rebuild the public datasets:

- `scripts/prepare_exchange_rate_dataset.py` (Exchange, GluonTS)
- `scripts/prepare_fx30_ecb_dataset.py` (FX29, ECB API, use `--exclude-usd`)
- `scripts/prepare_industry49_dataset.py` (Ken French 49 Industry)
- `scripts/prepare_ishares_dataset.py` (iShares ETFs; place CSVs under `ishares/`)

## 3. Training and Sampling
The notebook `notebooks/reproduce_thesis.ipynb` lists the commands used for the thesis experiments. Key examples:

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
Use the overrides listed in `reproduce_thesis.ipynb` to match the thesis (context/pred lengths, Beta t-sampling, 1000 EM steps, etc.).

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

## 5. Plotting / Metrics
See `notebooks/reproduce_thesis.ipynb` or `notebooks/paper.ipynb` for the exact commands. Common scripts include:

- `scripts/plot_stylized_facts.py`
- `scripts/plot_bull_bear_regimes.py`
- `scripts/plot_dataset_spectrum.py`
- `scripts/plot_eigen_windows.py`
- `scripts/plot_asset_panel.py`
- `scripts/plot_pair_window_metric.py`
- `scripts/plot_correlation_table.py`
- `scripts/make_dataset_summary.py`, `scripts/make_window_count_table.py`
- `scripts/make_paper_metric_tables.py`, `scripts/compute_loglik_regret.py`, `scripts/make_ablation_context_table.py`
- `scripts/compute_cross_signal_energy.py`, `scripts/plot_paper_figures.sh`

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
