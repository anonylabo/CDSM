#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${HOME}/.gluonts/datasets"

run() {
  printf '\n$ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

cd "$ROOT"

### Stylized facts #############################################################
run python scripts/plot_stylized_facts.py \
  --data-dirs "${DATA_ROOT}/exchange_rate_clean" \
  --split train \
  --max-assets 8 \
  --acf-lags 50 \
  --rolling-window 60 \
  --full-series-mode cum_returns \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_stylized_facts.py \
  --data-dirs "${DATA_ROOT}/fx30_ecb" \
  --split train \
  --max-assets 29 \
  --acf-lags 50 \
  --rolling-window 60 \
  --full-series-mode cum_returns \
  --exclude-assets TRY \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_stylized_facts.py \
  --data-dirs "${DATA_ROOT}/industry49_clean" \
  --split train \
  --max-assets 49 \
  --acf-lags 50 \
  --full-series-mode cum_returns \
  --rolling-window 60 \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_stylized_facts.py \
  --data-dirs "${DATA_ROOT}/industry49_clean_0.85" \
  --split train \
  --max-assets 49 \
  --acf-lags 50 \
  --full-series-mode cum_returns \
  --rolling-window 60 \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_stylized_facts.py \
  --data-dirs "${DATA_ROOT}/ishares14_clean" \
  --split train \
  --max-assets 14 \
  --acf-lags 50 \
  --full-series-mode cum_returns \
  --rolling-window 60 \
  --output-dir assets \
  --output-format pdf

### Bull/bear regimes ##########################################################
run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/exchange_rate_clean" \
  --split all --max-assets 8 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/fx30_ecb" \
  --split all --max-assets 29 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/industry49_clean" \
  --split all --max-assets 49 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/industry49_clean_0.85" \
  --split all --max-assets 49 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/ishares14_clean" \
  --split all --max-assets 49 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf

run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/fx30_ecb" \
  --split all --max-assets 29 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf
run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/industry49_clean" \
  --split all --max-assets 49 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf
run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/industry49_clean_0.85" \
  --split all --max-assets 49 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf
run python scripts/plot_bull_bear_regimes.py \
  --data-dirs "${DATA_ROOT}/ishares14_clean" \
  --split all --max-assets 14 \
  --regime-window 60 --regime-threshold 0 \
  --split-boundary-label "Train/Test" \
  --output-dir assets \
  --output-format pdf
run python scripts/combine_images.py \
  --input "FX30:assets/bull_bear_fx30_ecb_all.png" \
  --input "Industry49:assets/bull_bear_industry49_clean_all.png" \
  --input "Industry49:assets/bull_bear_industry49_clean_0.85_all.png" \
  --input "iShares14:assets/bull_bear_ishares14_clean_all.png" \
  --layout vertical \
  --no-labels \
  --output assets/bull_bear_all_combined.png

### Spectrum #############################################################
run env PYTHONPATH=src python scripts/plot_dataset_spectrum.py \
  --data-dirs \
    "${DATA_ROOT}/exchange_rate_clean" \
    "${DATA_ROOT}/fx30_ecb" \
    "${DATA_ROOT}/industry49_clean" \
    "${DATA_ROOT}/industry49_clean_0.85" \
    "${DATA_ROOT}/ishares14_clean" \
  --split train \
  --combined-stack-datasets \
  --combined-output assets/spectrum_cmp.png \
  --per-dataset-dir assets/spectrum_per_dataset \
  --output-format pdf

### Eigen windows ########################################################
# Optional dependency (used when combining PDF outputs):
# run pip install pymupdf
run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/exchange_rate_clean" --split train \
  --context-len 30 --pred-len 15 --val-ratio 0.3 \
  --regime-manifest assets/paper/FX8/window_regimes_fx8_train_30_15val0.3train0.8.json \
  --regime-stats assets/paper/FX8/window_regimes_fx8_train_stats_30_15val0.3train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_fx8_train_30_15.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/fx30_ecb" --split train \
  --context-len 45 --pred-len 15 --val-ratio 0.3 \
  --regime-manifest assets/paper/FX30/FX30_45_15val0.3train0.8/window_regimes_fx30_ecb_train_45_15val0.3train0.8.json \
  --regime-stats assets/paper/FX30/FX30_45_15val0.3train0.8/window_regimes_fx30_ecb_train_stats_45_15val0.3train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_fx30_ecb_train_45_15.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/fx30_ecb" --split train \
  --context-len 60 --pred-len 30 --val-ratio 0.2 \
  --regime-manifest assets/paper/FX30/FX30_ecb_60_30val0.2train0.8/window_regimes_fx30_ecb_train_60_30val0.2train0.8.json \
  --regime-stats assets/paper/FX30/FX30_ecb_60_30val0.2train0.8/window_regimes_fx30_ecb_train_stats_60_30val0.2train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_fx30_ecb_train_60_30.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/industry49_clean" --split train \
  --context-len 30 --pred-len 15 --val-ratio 0.3 \
  --regime-manifest assets/paper/Industry49/ind49_30_15val0.3train0.8/window_regimes_industry49_clean_train_30_15val0.3train0.8.json \
  --regime-stats assets/paper/Industry49/ind49_30_15val0.3train0.8/window_regimes_industry49_clean_train_stats_30_15val0.3train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_ind49_train_30_15.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/industry49_clean" --split train \
  --context-len 40 --pred-len 20 --val-ratio 0.3 \
  --regime-manifest assets/paper/Industry49/ind49_40_20val0.3train0.8/window_regimes_industry49_clean_train_40_20val0.3train0.8.json \
  --regime-stats assets/paper/Industry49/ind49_40_20val0.3train0.8/window_regimes_industry49_clean_train_stats_40_20val0.3train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_ind49_train_40_20.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/industry49_clean_0.85" --split train \
  --context-len 80 --pred-len 40 --val-ratio -0.05 \
  --regime-manifest assets/paper/Industry49/ind49_80_40val-0.05train0.85/window_regimes_industry49_clean_train_80_40val-0.05train0.85.json \
  --regime-stats assets/paper/Industry49/ind49_80_40val-0.05train0.85/window_regimes_industry49_clean_train_stats_80_40val-0.05train0.85.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_ind49_train_80_40.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/ishares14_clean" --split train \
  --context-len 30 --pred-len 15 --val-ratio 0.3 \
  --regime-manifest assets/paper/ishares14/ishares14_30_15_val0.3train0.8/window_regimes_ishares14_clean_train_30_15_val0.3train0.8.json \
  --regime-stats assets/paper/ishares14/ishares14_30_15_val0.3train0.8/window_regimes_ishares14_clean_train_30_15_val0.3train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_ishares14_train_30_15.pdf"

run env PYTHONPATH=src python scripts/plot_eigen_windows.py \
  --dataset "${DATA_ROOT}/ishares14_clean" --split train \
  --context-len 100 --pred-len 20 --val-ratio -0.05 \
  --regime-manifest assets/paper/ishares14/ishares14_100_20_val-0.05train0.8/window_regimes_ishares14_clean_train_100_20_val-0.05train0.8.json \
  --regime-stats assets/paper/ishares14/ishares14_100_20_val-0.05train0.8/window_regimes_ishares14_clean_train_stats_100_20_val-0.05train0.8.json \
  --regime-alpha 0.25 \
  --output-format pdf \
  --output "assets/eigen_ishares14_train_100_20.pdf"

run env PYTHONPATH=src python scripts/combine_images.py \
  --input "FX8_train_30_15:assets/eigen_fx8_train_30_15.pdf" \
  --input "FX30_ecb_train_45_15:assets/eigen_fx30_ecb_train_45_15.pdf" \
  --input "FX30_ecb_train_60_30:assets/eigen_fx30_ecb_train_60_30.pdf" \
  --input "ind49_30_15:assets/eigen_ind49_train_30_15.pdf" \
  --input "ind49_40_20:assets/eigen_ind49_train_40_20.pdf" \
  --input "ind49_80_40:assets/eigen_ind49_train_80_40.pdf" \
  --input "ishares14_100_20:assets/eigen_ishares14_train_100_20.pdf" \
  --layout vertical \
  --output-format pdf \
  --output "assets/eigen_combined_train.pdf"

### Dataset summary & window counts ###########################################
run python scripts/make_dataset_summary.py \
  --entry "FX8|${DATA_ROOT}/exchange_rate_clean|30|15|0.3|GluonTS Exchange Rate" \
  --entry "FX30|${DATA_ROOT}/fx30_ecb|45|15|0.3|ECB Exchange Rate" \
  --entry "FX30|${DATA_ROOT}/fx30_ecb|60|30|0.2|ECB Exchange Rate" \
  --entry "ind49|${DATA_ROOT}/industry49_clean|train|30|15|0.3|Ken French Industry 49" \
  --entry "ind49|${DATA_ROOT}/industry49_clean|train|40|20|0.3|Ken French Industry 49" \
  --entry "ind49|${DATA_ROOT}/industry49_clean_0.85|train|80|40|-0.05|Ken French Industry 49" \
  --entry "ishares14|${DATA_ROOT}/ishares14_clean|train|30|15|0.3|iShares ETF panel" \
  --entry "ishares14|${DATA_ROOT}/ishares14_clean|train|100|20|-0.05|iShares ETF panel" \
  --infer-period-override \
  --latex-tight \
  --output-markdown assets/dataset_summary.md \
  --output-latex assets/dataset_summary.tex \
  --per-entry-markdown-dir assets/dataset_summary_per_entry \
  --per-entry-latex-dir assets/dataset_summary_per_entry

run python scripts/make_window_count_table.py \
  --dataset "FX8:${DATA_ROOT}/exchange_rate_clean:30:15:0.3:0.8:1" \
  --dataset "FX30:${DATA_ROOT}/fx30_ecb:45:15:0.3:0.8:1" \
  --dataset "FX30:${DATA_ROOT}/fx30_ecb:60:30:0.2:0.8:1" \
  --dataset "ind49:${DATA_ROOT}/industry49_clean:30:15:0.3:0.8:1" \
  --dataset "ind49:${DATA_ROOT}/industry49_clean:40:20:0.3:0.8:1" \
  --dataset "ind49:${DATA_ROOT}/industry49_clean_0.85:80:40:-0.05:0.85:1" \
  --dataset "ishares14:${DATA_ROOT}/ishares14_clean:30:15:0.3:0.8:1" \
  --dataset "ishares14:${DATA_ROOT}/ishares14_clean:100:20:-0.05:0.8:1" \
  --latex-tight \
  --no-apply-train-ratio \
  --train-ratio-from-metadata \
  --output-markdown assets/window_counts.md \
  --output-latex assets/window_counts.tex \
  --per-entry-latex-dir assets/window_counts_per_entry

run python scripts/make_data_overview_table.py \
  --dataset-summary assets/dataset_summary.tex \
  --window-counts assets/window_counts.tex \
  --out assets/data_overview.tex

### MC trajectory selection (single path) #####################################
run python scripts/select_mc_trajectory.py \
  --mc-index 0 \
  --overwrite \
  --batch-dir outputs/time/conditional/20260110-060824_FX8_30_15val0.3train0.8/samples_history/batch-20260110-091835 \
  --batch-dir outputs/time/unconditional/20260110-060945_FX8_30_15val0.3train0.8/samples_history/batch-20260110-092049 \
  --batch-dir outputs/fourier/conditional/20260110-060905_FX8_30_15val0.3train0.8/samples_history/batch-20260110-091949 \
  --batch-dir outputs/fourier/unconditional/20260110-061031_FX8_30_15val0.3train0.8/samples_history/batch-20260110-092221 \
  --batch-dir outputs/time/conditional/20251219-214457_FX30_45_15val0.3train0.8/samples_history/batch-20251220-033209 \
  --batch-dir outputs/time/unconditional/20251219-214525_FX30_45_15val0.3train0.8/samples_history/batch-20251220-000143 \
  --batch-dir outputs/fourier/conditional/20251219-214513_FX30_45_15val0.3train0.8/samples_history/batch-20251220-033317 \
  --batch-dir outputs/fourier/unconditional/20251219-214538_FX30_45_15val0.3train0.8/samples_history/batch-20251220-000257 \
  --batch-dir outputs/time/conditional/20251222-191349_FX30_60_30val0.2train0.8/samples_history/batch-20251222-214223 \
  --batch-dir outputs/time/unconditional/20251222-184829_FX30_60_30val0.2train0.8/samples_history/batch-20251222-214430 \
  --batch-dir outputs/fourier/conditional/20251222-154441_FX30_60_30val0.2train0.8/samples_history/batch-20251222-214325 \
  --batch-dir outputs/fourier/unconditional/20251222-184846_FX30_60_30val0.2train0.8/samples_history/batch-20251222-214532 \
  --batch-dir outputs/time/conditional/20251221-160141_ind49_40_20val0.3train0.8/samples_history/batch-20251221-202751 \
  --batch-dir outputs/time/unconditional/20251221-160224_ind49_40_20val0.3train0.8/samples_history/batch-20251221-203025 \
  --batch-dir outputs/fourier/conditional/20251221-160201_ind49_40_20val0.3train0.8/samples_history/batch-20251221-202856 \
  --batch-dir outputs/fourier/unconditional/20251221-160254_ind49_40_20val0.3train0.8/samples_history/batch-20251221-203202 \
  --batch-dir outputs/time/conditional/20251223-082139_ind49_80_40val-0.05train0.85/samples_history/batch-20251223-210923 \
  --batch-dir outputs/time/unconditional/20251223-082452_ind49_80_40val-0.05train0.85/samples_history/batch-20251223-231519 \
  --batch-dir outputs/fourier/conditional/20251223-082426_ind49_80_40val-0.05train0.85/samples_history/batch-20251223-211029 \
  --batch-dir outputs/fourier/unconditional/20251223-082520_ind49_80_40val-0.05train0.85/samples_history/batch-20251223-231628 \
  --batch-dir outputs/time/conditional/20251224-190527_ind49_30_15val0.3train0.8/samples_history/batch-20251224-231817 \
  --batch-dir outputs/time/unconditional/20251224-190627_ind49_30_15val0.3train0.8/samples_history/batch-20251224-231833 \
  --batch-dir outputs/fourier/conditional/20251224-190554_ind49_30_15val0.3train0.8/samples_history/batch-20251224-231801 \
  --batch-dir outputs/fourier/unconditional/20251224-190659_ind49_30_15val0.3train0.8/samples_history/batch-20251224-231903 \
  --batch-dir outputs/time/conditional/20251225-164921_ishares14_30_15_val0.3train0.8/samples_history/batch-20251225-214426 \
  --batch-dir outputs/time/unconditional/20251225-165041_ishares14_30_15_val0.3train0.8/samples_history/batch-20251225-214612 \
  --batch-dir outputs/fourier/conditional/20251225-165001_ishares14_30_15_val0.3train0.8/samples_history/batch-20251226-011341 \
  --batch-dir outputs/fourier/unconditional/20251225-165213_ishares14_30_15_val0.3train0.8/samples_history/batch-20251225-214657 \
  --batch-dir outputs/time/conditional/20251220-184214_ishares14_100_20_val-0.05train0.8/samples_history/batch-20251220-215853 \
  --batch-dir outputs/time/unconditional/20251220-184302_ishares14_100_20_val-0.05train0.8/samples_history/batch-20251220-215125 \
  --batch-dir outputs/fourier/conditional/20251220-184240_ishares14_100_20_val-0.05train0.8/samples_history/batch-20251220-215750 \
  --batch-dir outputs/fourier/unconditional/20251220-184317_ishares14_100_20_val-0.05train0.8/samples_history/batch-20251220-215407 \
  --batch-dir outputs/time/conditional/20260129-223951_FX8_30_15val0.3train0.8_lambda0/samples_history/batch-20260130-075051 \
  --batch-dir outputs/fourier/conditional/20260129-224014_FX8_30_15val0.3train0.8_lambda0/samples_history/batch-20260130-075153 \
  --batch-dir outputs/time/conditional/20260129-224047_FX30_ecb_45_15val0.3train0.8_lambda0/samples_history/batch-20260130-075313 \
  --batch-dir outputs/fourier/conditional/20260129-224103_FX30_ecb_45_15val0.3train0.8_lambda0/samples_history/batch-20260130-075442 \
  --batch-dir outputs/time/conditional/20260129-224207_ind49_30_15val0.3train0.8_lambda0/samples_history/batch-20260130-075618 \
  --batch-dir outputs/fourier/conditional/20260129-224232_ind49_30_15val0.3train0.8_lambda0/samples_history/batch-20260130-075727 \
  --batch-dir outputs/time/conditional/20260129-224313_ishares14_30_15_val0.3train0.8_lambda0/samples_history/batch-20260130-075904 \
  --batch-dir outputs/fourier/conditional/20260129-224344_ishares14_30_15_val0.3train0.8_lambda0/samples_history/batch-20260130-080017 \
  --batch-dir outputs/time/conditional/20260201-064735_FX8_30_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260201-194302 \
  --batch-dir outputs/time/conditional/20260201-064843_FX8_30_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260201-194518 \
  --batch-dir outputs/time/conditional/20260201-064936_FX8_30_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260201-194755 \
  --batch-dir outputs/fourier/conditional/20260201-064803_FX8_30_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260201-194401 \
  --batch-dir outputs/fourier/conditional/20260201-064905_FX8_30_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260201-194640 \
  --batch-dir outputs/fourier/conditional/20260201-064951_FX8_30_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260201-194903 \
  --batch-dir outputs/time/conditional/20260201-065046_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260201-195024 \
  --batch-dir outputs/time/conditional/20260201-065152_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260201-195310 \
  --batch-dir outputs/time/conditional/20260201-065234_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260201-195644 \
  --batch-dir outputs/fourier/conditional/20260201-065102_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260201-195140 \
  --batch-dir outputs/fourier/conditional/20260201-065207_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260201-195451 \
  --batch-dir outputs/fourier/conditional/20260201-065253_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260201-195740 \
  --batch-dir outputs_local/time/conditional/20260202-160436_ind49_30_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260203-064607 \
  --batch-dir outputs_local/time/conditional/20260202-160514_ind49_30_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260203-064836 \
  --batch-dir outputs_local/time/conditional/20260202-160558_ind49_30_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260203-065039 \
  --batch-dir outputs_local/fourier/conditional/20260202-160447_ind49_30_15val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260203-064719 \
  --batch-dir outputs_local/fourier/conditional/20260202-160537_ind49_30_15val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260203-064925 \
  --batch-dir outputs_local/fourier/conditional/20260202-160612_ind49_30_15val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260203-065132 \
  --batch-dir outputs_local/time/conditional/20260202-160654_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260203-065303 \
  --batch-dir outputs_local/time/conditional/20260202-160723_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260203-065538 \
  --batch-dir outputs_local/time/conditional/20260202-160758_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260203-065800 \
  --batch-dir outputs_local/fourier/conditional/20260202-160704_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed12/samples_history/batch-20260203-065414 \
  --batch-dir outputs_local/fourier/conditional/20260202-160734_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed22/samples_history/batch-20260203-065644 \
  --batch-dir outputs_local/fourier/conditional/20260202-160808_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed32/samples_history/batch-20260203-065855

### Correlation tables #########################################################
run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --pivot-regime \
  --only-pred-truth \
  --sig-digits 4 \
  --regime-manifest "${ROOT}/assets/paper/FX8/window_regimes_fx8_test_30_15val0.3train0.8.json" \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20260110-060824_FX8_30_15val0.3train0.8" \
  --augmented-run-substr "outputs/fourier/conditional/20260110-060905_FX8_30_15val0.3train0.8" \
  --asset-offset 0 --assets 8 \
  --runs \
    "outputs/time/conditional/20260110-060824_FX8_30_15val0.3train0.8" \
    "outputs/time/unconditional/20260110-060945_FX8_30_15val0.3train0.8" \
    "outputs/fourier/conditional/20260110-060905_FX8_30_15val0.3train0.8" \
    "outputs/fourier/unconditional/20260110-061031_FX8_30_15val0.3train0.8" \
  --sample-tags \
    batch-20260110-091835-mc0 \
    batch-20260110-092049-mc0 \
    batch-20260110-091949-mc0 \
    batch-20260110-092221-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20260110-125308_FX8_30_15val0.3train0.8|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20260110-125253_FX8_30_15val0.3train0.8|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20260110-130854_dcc_garch_cov_FX8_30_15val0.3train0.8|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20260110-130842_dcc_garch_corr_FX8_30_15val0.3train0.8|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20260110-132330_FX8_30_15val0.3train0.8|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20260110-125326_FX8_30_15val0.3train0.8|deepvar|DeepVAR" \
  --out-md assets/table_FX8_30_15val0.3train0.8.md \
  --out-png assets/table_FX8_30_15val0.3train0.8.png \
  --out-tex assets/table_FX8_30_15val0.3train0.8.tex \
  --out-pdf assets/table_FX8_30_15val0.3train0.8.pdf

run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --pivot-regime \
  --only-pred-truth \
  --sig-digits 4 \
  --regime-manifest "${ROOT}/assets/paper/FX30/FX30_45_15val0.3train0.8/window_regimes_fx30_ecb_test_45_15val0.3train0.8.json" \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20251219-214457_FX30_45_15val0.3train0.8" \
  --augmented-run-substr "outputs/fourier/conditional/20251219-214513_FX30_45_15val0.3train0.8" \
  --asset-offset 0 --assets 29 \
  --runs \
    "outputs/time/conditional/20251219-214457_FX30_45_15val0.3train0.8" \
    "outputs/time/unconditional/20251219-214525_FX30_45_15val0.3train0.8" \
    "outputs/fourier/conditional/20251219-214513_FX30_45_15val0.3train0.8" \
    "outputs/fourier/unconditional/20251219-214538_FX30_45_15val0.3train0.8" \
  --sample-tags \
    batch-20251220-033209-mc0 \
    batch-20251220-000143-mc0 \
    batch-20251220-033317-mc0 \
    batch-20251220-000257-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20251223-111855_FX30_45_15val0.3train0.8|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20251220-080604_FX30_45_15val0.3train0.8|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-135709_dcc_garch_cov_FX30_45_15val0.3train0.8|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-135604_dcc_garch_corr_FX30_45_15val0.3train0.8|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20251223-134444_FX30_45_15val0.3train0.8|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-154042_FX30_45_15val0.3train0.8|deepvar|DeepVAR" \
  --out-md assets/table_FX30_45_15val0.3train0.8.md \
  --out-png assets/table_FX30_45_15val0.3train0.8.png \
  --out-tex assets/table_FX30_45_15val0.3train0.8.tex

run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --pivot-regime \
  --only-pred-truth \
  --sig-digits 4 \
  --regime-manifest "${ROOT}/assets/paper/FX30/FX30_ecb_60_30val0.2train0.8/window_regimes_fx30_ecb_test_60_30val0.2train0.8.json" \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20251222-191349_FX30_60_30val0.2train0.8" \
  --augmented-run-substr "outputs/fourier/conditional/20251222-154441_FX30_60_30val0.2train0.8" \
  --asset-offset 0 --assets 29 \
  --runs \
    "outputs/time/conditional/20251222-191349_FX30_60_30val0.2train0.8" \
    "outputs/time/unconditional/20251222-184829_FX30_60_30val0.2train0.8" \
    "outputs/fourier/conditional/20251222-154441_FX30_60_30val0.2train0.8" \
    "outputs/fourier/unconditional/20251222-184846_FX30_60_30val0.2train0.8" \
  --sample-tags \
    batch-20251222-214223-mc0 \
    batch-20251222-214430-mc0 \
    batch-20251222-214325-mc0 \
    batch-20251222-214532-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20251223-111941_FX30_60_30val0.2train0.8|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20251222-134331_FX30_60_30val0.2train0.8|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-141424_dcc_garch_cov_FX30_60_30val0.2train0.8|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-141416_dcc_garch_corr_FX30_60_30val0.2train0.8|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20251223-140033_FX30_60_30val0.2train0.8|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251222-214609_FX30_60_30val0.2train0.8|deepvar|DeepVAR" \
  --out-md assets/table_FX30_60_30val0.2train0.8.md \
  --out-png assets/table_FX30_60_30val0.2train0.8.png \
  --out-tex assets/table_FX30_60_30val0.2train0.8.tex

run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --only-pred-truth \
  --pivot-regime \
  --sig-digits 4 \
  --regime-manifest "${ROOT}/assets/paper/Industry49/ind49_40_20val0.3train0.8/window_regimes_industry49_clean_test_40_20val0.3train0.8.json" \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20251221-160141_ind49_40_20val0.3train0.8" \
  --augmented-run-substr "outputs/fourier/conditional/20251221-160201_ind49_40_20val0.3train0.8" \
  --asset-offset 0 --assets 49 \
  --runs \
    "outputs/time/conditional/20251221-160141_ind49_40_20val0.3train0.8" \
    "outputs/time/unconditional/20251221-160224_ind49_40_20val0.3train0.8" \
    "outputs/fourier/conditional/20251221-160201_ind49_40_20val0.3train0.8" \
    "outputs/fourier/unconditional/20251221-160254_ind49_40_20val0.3train0.8" \
  --sample-tags \
    batch-20251221-202751-mc0 \
    batch-20251221-203025-mc0 \
    batch-20251221-202856-mc0 \
    batch-20251221-203202-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20251223-112029_ind49_40_20val0.3train0.8|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20251221-201915_ind49_40_20val0.3train0.8|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-144212_dcc_garch_cov_ind49_40_20val0.3train0.8|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-144158_dcc_garch_corr_ind49_40_20val0.3train0.8|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20251223-141701_ind49_40_20val0.3train0.8|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-225650_ind49_40_20val0.3train0.8|deepvar|DeepVAR" \
  --out-md assets/table_ind49_40_20val0.3train0.8.md \
  --out-png assets/table_ind49_40_20val0.3train0.8.png \
  --out-tex assets/table_ind49_40_20val0.3train0.8.tex

run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --only-pred-truth \
  --pivot-regime \
  --sig-digits 4 \
  --regime-manifest "${ROOT}/assets/paper/Industry49/ind49_80_40val-0.05train0.85/window_regimes_industry49_clean_test_80_40val-0.05train0.85.json" \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20251223-082139_ind49_80_40val-0.05train0.85" \
  --augmented-run-substr "outputs/fourier/conditional/20251223-082426_ind49_80_40val-0.05train0.85" \
  --asset-offset 0 --assets 49 \
  --runs \
    "outputs/time/conditional/20251223-082139_ind49_80_40val-0.05train0.85" \
    "outputs/time/unconditional/20251223-082452_ind49_80_40val-0.05train0.85" \
    "outputs/fourier/conditional/20251223-082426_ind49_80_40val-0.05train0.85" \
    "outputs/fourier/unconditional/20251223-082520_ind49_80_40val-0.05train0.85" \
  --sample-tags \
    batch-20251223-210923-mc0 \
    batch-20251223-231519-mc0 \
    batch-20251223-211029-mc0 \
    batch-20251223-231628-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20251223-112136_ind49_80_40val-0.05train0.85|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20251223-091502_ind49_80_40val-0.05train0.85|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-205430_dcc_garch_cov_ind49_80_40val-0.05train0.85|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-205422_dcc_garch_corr_ind49_80_40val-0.05train0.85|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20251223-193333_ind49_80_40val-0.05train0.85|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251223-200254_ind49_80_40val-0.05train0.85|deepvar|DeepVAR" \
  --out-md assets/table_ind49_80_40val-0.05train0.85.md \
  --out-png assets/table_ind49_80_40val-0.05train0.85.png \
  --out-tex assets/table_ind49_80_40val-0.05train0.85.tex

run python scripts/plot_correlation_table.py \
  --batch-aggregate \
  --pivot-regime \
  --regime-manifest "${ROOT}/assets/paper/ishares14/ishares14_100_20_val-0.05train0.8/window_regimes_ishares14_clean_test_100_20_val-0.05train0.8.json" \
  --only-pred-truth \
  --sig-digits 4 \
  --metric-name matrix_cov_fro --matrix-kind cov \
  --metric-name matrix_corr_fro --matrix-kind corr \
  --metric-name corr_wasserstein --matrix-kind corr \
  --metric-name eigen_wasserstein --matrix-kind corr \
  --augmented-run-substr "outputs/time/conditional/20251220-184214_ishares14_100_20_val-0.05train0.8" \
  --augmented-run-substr "outputs/fourier/conditional/20251220-184240_ishares14_100_20_val-0.05train0.8" \
  --asset-offset 0 --assets 14 \
  --runs \
    "outputs/time/conditional/20251220-184214_ishares14_100_20_val-0.05train0.8" \
    "outputs/time/unconditional/20251220-184302_ishares14_100_20_val-0.05train0.8" \
    "outputs/fourier/conditional/20251220-184240_ishares14_100_20_val-0.05train0.8" \
    "outputs/fourier/unconditional/20251220-184317_ishares14_100_20_val-0.05train0.8" \
  --sample-tags \
    batch-20251220-215853-mc0 \
    batch-20251220-215125-mc0 \
    batch-20251220-215750-mc0 \
    batch-20251220-215407-mc0 \
  --stage test \
  --baseline "outputs/baselines/window_uncond/conditional/batch-20251223-112215_ishares14_100_20_val-0.05train0.8|window_uncond|Global Static Empirical Covariance" \
  --baseline "outputs/baselines/window_context/conditional/batch-20251221-083937_ishares14_100_20_val-0.05train0.8|window_context|Sample Covariance" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-193120_dcc_garch_cov_ishares14_100_20_val-0.05train0.8|dcc_garch_cov|DCC cov|cov" \
  --baseline "outputs/baselines/dcc_garch/conditional/batch-20251223-193133_dcc_garch_corr_ishares14_100_20_val-0.05train0.8|dcc_garch_corr|DCC corr|corr" \
  --baseline "outputs/baselines/cab/conditional/batch-20251223-144801_ishares14_100_20_val-0.05train0.8|cab|CAB" \
  --baseline "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-103122_ishares14_100_20_val-0.05train0.8|deepvar|DeepVAR" \
  --out-md assets/table_ishares14_100_20_val-0.05train0.8.md \
  --out-png assets/table_ishares14_100_20_val-0.05train0.8.png \
  --out-tex assets/table_ishares14_100_20_val-0.05train0.8.tex

### Seed robustness tables (lambda=0) ######################################
run python scripts/make_seed_robustness_tables.py \
  --sig-digits 4 \
  --stage all \
  --out-cov assets/table_seed_robustness_cov_lambda0.tex \
  --out-corr assets/table_seed_robustness_corr_lambda0.tex

### Asset panels ###############################################################
run python scripts/plot_asset_panel.py \
  --models \
    "Fourier-Cond|outputs/fourier/conditional/20251219-214513_FX30_45_15val0.3train0.8" \
    "Fourier-Uncond|outputs/fourier/unconditional/20251219-214538_FX30_45_15val0.3train0.8" \
    "Time-Cond|outputs/time/conditional/20251219-214457_FX30_45_15val0.3train0.8" \
    "Time-Uncond|outputs/time/unconditional/20251219-214525_FX30_45_15val0.3train0.8" \
    "DeepVAR|outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-154042_FX30_45_15val0.3train0.8" \
  --windows 80 120 160 180 \
  --asset-ids CNY JPY GBP CHF AUD TRY \
  --panel-size 3.5 2.2 \
  --overlay-models \
  --batch-aggregate \
  --dataset "${DATA_ROOT}/fx30_ecb" \
  --output assets/panel_cmp.png

run python scripts/plot_asset_panel.py \
  --models \
    "Fourier-Cond|outputs/fourier/conditional/20251221-160201_ind49_40_20val0.3train0.8" \
    "Fourier-Uncond|outputs/fourier/unconditional/20251221-160254_ind49_40_20val0.3train0.8" \
    "Time-Cond|outputs/time/conditional/20251221-160141_ind49_40_20val0.3train0.8" \
    "Time-Uncond|outputs/time/unconditional/20251221-160224_ind49_40_20val0.3train0.8" \
    "DeepVAR|outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-225650_ind49_40_20val0.3train0.8" \
  --windows 80 120 160 180 \
  --panel-size 3.5 2.2 \
  --overlay-models \
  --batch-aggregate \
  --dataset "${DATA_ROOT}/industry49_clean" \
  --asset-ids Agric Util Oil Banks Chips Rtail Steel Gold \
  --output assets/panel_cmp.png

run python scripts/plot_asset_panel.py \
  --models \
    "Fourier-Cond|outputs/fourier/conditional/20251220-184240_ishares14_100_20_val-0.05train0.8" \
    "DeepVAR|outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251221-103122_ishares14_100_20_val-0.05train0.8" \
  --windows 40 60 80 100 120 128 \
  --panel-size 3.5 2.2 \
  --overlay-models \
  --batch-aggregate \
  --dataset "${DATA_ROOT}/ishares14_clean" \
  --asset-ids AGG IAGG IXC IXG IXJ IXN IYR JXI \
  --output assets/panel_cmp.png

### Series metrics table #######################################################
run python scripts/make_paper_metric_tables.py \
  --metrics series_crps series_nd series_rmse \
  --batch-aggregate \
  --table-tex assets/table_series.tex \
  --time-conditional "outputs/time/conditional/20251204-122500_data4_train0.8_val-0.1(100,20)" \
  --fourier-conditional "outputs/fourier/conditional/20251204-182351_data4_train0.8_val-0.1(100,20)" \
  --time-unconditional "outputs/time/unconditional/20251205-013549_data4_train0.8_val-0.1(100,20)" \
  --fourier-unconditional "outputs/fourier/unconditional/20251205-065047_data4_train0.8_val-0.1(100,20)" \
  --include-deepvar \
  --deepvar-path "outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251205-081657_data4_train0.8_val-0.1(100,20)" \
  --use-samples-history \
  --samples-history "Time (Conditional)=outputs/time/conditional/20251204-122500_data4_train0.8_val-0.1(100,20)/samples_history/batch-20251204-152935" \
  --samples-history "Fourier (Conditional)=outputs/fourier/conditional/20251204-182351_data4_train0.8_val-0.1(100,20)/samples_history/batch-20251204-232343" \
  --samples-history "Time (Unconditional)=outputs/time/unconditional/20251205-013549_data4_train0.8_val-0.1(100,20)/samples_history/batch-20251205-060646" \
  --samples-history "Fourier (Unconditional)=outputs/fourier/unconditional/20251205-065047_data4_train0.8_val-0.1(100,20)/samples_history/batch-20251205-074000" \
  --stage-splits

### Loss composition ###########################################################
run python scripts/plot_loss_composition_total.py \
  "${ROOT}/outputs/time/conditional" \
  "${ROOT}/outputs/fourier/conditional" \
  --lambda 5e-4 \
  --out "${ROOT}/assets/loss_composition_by_dataset_lambda5e-4.pdf" \
  --exclude FX8

run python scripts/plot_loss_composition_total.py \
  "${ROOT}/outputs/time/conditional" \
  "${ROOT}/outputs/fourier/conditional" \
  --lambda 5e-3 \
  --out "${ROOT}/assets/loss_composition_by_dataset_lambda5e-3.pdf" \
  --exclude FX8

### Correlation trend (combined) ##############################################
BLOCK_FX_30_15=$(cat <<'EOF'
label: fx_30_15
stage: test
asset_counts: "1..8"
runs: [outputs/time/conditional/20251126-064815_val0.3,
       outputs/time/unconditional/20251126-143145_val0.3,
       outputs/fourier/conditional/20251126-101856_val0.3,
       outputs/fourier/unconditional/20251126-152536_val0.3]
sample_tags: [batch-20251128-193550, batch-20251128-203816, batch-20251128-105736, batch-20251128-122637]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163144_data1_val0.3|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251128-213431_data1_val0.3|window_local|Expanding Window
  - outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251128-210211_data1_val0.3|deepvar|DeepVAR
  - outputs/baselines/cab/conditional/batch-20251129-042413_data1_val0.3|cab|CAB
EOF
)
BLOCK_IND49_30_15=$(cat <<'EOF'
label: ind49_30_15
stage: test
asset_counts: "1..14"
runs: [outputs/time/conditional/20251129-050942_data2_val0.3,
       outputs/time/unconditional/20251129-145406_data2_val0.3,
       outputs/fourier/conditional/20251129-102706_data2_val0.3,
       outputs/fourier/unconditional/20251129-163157_data2_val0.3]
sample_tags: [batch-20251129-083615, batch-20251129-155730, batch-20251129-132516, batch-20251129-172237]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163309_data2_val0.3|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251129-175326_data2_val0.3|window_local|Expanding Window
  - outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251129-175417_data2_val0.3|deepvar|DeepVAR
  - outputs/baselines/cab/conditional/batch-20251129-184041_data2_val0.3|cab|CAB
EOF
)
BLOCK_IND49_60_30=$(cat <<'EOF'
label: ind49_60_30
stage: test
asset_counts: "1..16"
runs:
  - outputs/time/conditional/20251129-195934_data2_val0.3(60,30)
  - outputs/time/unconditional/20251130-071659_data2_val0.3(60,30)
  - outputs/fourier/conditional/20251130-012441_data2_val0.3(60,30)
  - outputs/fourier/unconditional/20251130-082517_data2_val0.3(60,30)
sample_tags: [batch-20251129-220250, batch-20251130-080532, batch-20251130-061628, batch-20251130-090334]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163358_data2_val0.3(60,30)|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251130-091746_data2_val0.3(60,30)|window_local|Expanding Window
  - outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251130-091928_data2_val0.3(60,30)|deepvar|DeepVAR
  - outputs/baselines/cab/conditional/batch-20251130-102327_data2_val0.3(60,30)|cab|CAB
EOF
)
BLOCK_STOCK14_30_15=$(cat <<'EOF'
label: stock14_30_15
stage: test
asset_counts: "1..14"
runs:
  - outputs/time/conditional/20251203-002924_data3_val0.3(30,15)
  - outputs/time/unconditional/20251203-091602_data3_val0.3(30,15)
  - outputs/fourier/conditional/20251203-053707_data3_val0.3(30,15)
  - outputs/fourier/unconditional/20251203-081959_data3_val0.3(30,15)
sample_tags: [batch-20251203-044555, batch-20251203-095114, batch-20251203-073539, batch-20251203-085634]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163532_data3_val0.3(30,15)|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251203-101326_data3_val0.3(30,15)|window_local|Expanding Window
  - outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251203-102211_data3_val0.3(30,15)|deepvar|DeepVAR
  - outputs/baselines/cab/conditional/batch-20251203-105157_data3_val0.3(30,15)|cab|CAB
  - outputs/baselines/dcc_garch/conditional/batch-20251203-110402_data3_val0.3(30,15)|dcc_garch|DCC
EOF
)
BLOCK_STOCK6_100_20_T075=$(cat <<'EOF'
label: stock6_100_20_train0.75val0.3
stage: test
asset_counts: "1..6"
runs:
  - outputs/time/conditional/20251203-131933_data4_train0.75_val0.3(100,20)
  - outputs/time/unconditional/20251203-210235_data4_train0.75_val0.3(100,20)
  - outputs/fourier/conditional/20251203-182306_data4_train0.75_val0.3(100,20)
  - outputs/fourier/unconditional/20251203-222108_data4_train0.75_val0.3(100,20)
sample_tags: [batch-20251203-151830, batch-20251203-213903, batch-20251203-202806, batch-20251204-093151]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163727_data4_train0.75_val0.3(100,20)|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251204-094040_data4_train0.75_val0.3(100,20)|window_local|Expanding Window
  - outputs/baselines/deepvar/conditional/fx_deepvar/batch-20251204-094121_data4_train0.75_val0.3(100,20)|deepvar|DeepVAR
  - outputs/baselines/cab/conditional/batch-20251204-103551_data4_train0.75_val0.3(100,20)|cab|CAB
  - outputs/baselines/dcc_garch/conditional/batch-20251204-105433_data4_train0.75_val0.3(100,20)|dcc_garch|DCC
EOF
)
BLOCK_STOCK6_100_20_T08=$(cat <<'EOF'
label: stock6_100_20_train0.8val-0.1
stage: test
asset_counts: "1..6"
runs:
  - outputs/time/conditional/20251204-122500_data4_train0.8_val-0.1(100,20)
  - outputs/time/unconditional/20251205-013549_data4_train0.8_val-0.1(100,20)
  - outputs/fourier/conditional/20251204-182351_data4_train0.8_val-0.1(100,20)
  - outputs/fourier/unconditional/20251205-065047_data4_train0.8_val-0.1(100,20)
sample_tags: [batch-20251204-152935, batch-20251205-060646, batch-20251204-232343, batch-20251205-074000]
baseline:
  - outputs/baselines/window_uncond/conditional/batch-20251205-163910_data4_train0.8_val-0.1(100,20)|window_uncond|Global Static Empirical Covariance
  - outputs/baselines/window_local/conditional/batch-20251205-081146_data4_train0.8_val-0.1(100,20)|window_local|Expanding Window
  - outputs/baselines/cab/conditional/batch-20251205-084141_data4_train0.8_val-0.1(100,20)|cab|CAB
  - outputs/baselines/dcc_garch/conditional/batch-20251205-085554_data4_train0.8_val-0.1(100,20)|dcc_garch|DCC
EOF
)
run env PYTHONPATH=src python scripts/plot_correlation_trend.py \
  --metric-field pred_minus_truth \
  --batch-aggregate \
  --baseline-annotate last \
  --combined-png assets/corr_trend_combined.png \
  --out-png assets/corr_trend_all_agg.png \
  --block "$BLOCK_FX_30_15" \
  --block "$BLOCK_IND49_30_15" \
  --block "$BLOCK_IND49_60_30" \
  --block "$BLOCK_STOCK14_30_15" \
  --block "$BLOCK_STOCK6_100_20_T075" \
  --block "$BLOCK_STOCK6_100_20_T08"
