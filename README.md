# CollaFuse For Fraud Detection

This repo is now organized as a reproducible ICIS-style paper pipeline:

- `prepare-data`: raw fraud dataset -> standardized client splits
- `stage1-generate`: train CollaFuse and generate fraud-only synthetic samples with RQ1 fidelity artifacts
- `stage2-evaluate`: compare real-only, classical resampling, and generative fraud augmentation baselines with a small sklearn classifier suite
- `run-all-stages`: execute the full flow end-to-end

## Entry Point

Use the CLI:

```bash
python -m src.cli --config src/config_files/config_ieee_cis.yaml prepare-data
python -m src.cli --config src/config_files/config_ieee_cis.yaml stage1-generate
python -m src.cli --config src/config_files/config_ieee_cis.yaml stage2-evaluate
python -m src.cli --config src/config_files/config_ieee_cis.yaml run-all-stages
```

`stage2-evaluate` uses the latest Stage 1 run by default. You can still override it explicitly:

```bash
python -m src.cli --config src/config_files/config_ieee_cis.yaml stage2-evaluate --stage1-run <stage1_run_id>
```

`prepare-data` also accepts dataset-specific raw input overrides:

```bash
python -m src.cli --config <config> prepare-data --raw-main <path> --raw-aux <path> --raw-edge <path>
```

## Dataset Download

The IEEE-CIS config uses the original IEEE-CIS Fraud Detection training files from Kaggle:

- competition page: `https://www.kaggle.com/competitions/ieee-fraud-detection/data`
- required files:
  - `train_transaction.csv`
  - `train_identity.csv`

By default, the config expects them here:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Manual download:

1. Sign in to Kaggle and open the IEEE-CIS Fraud Detection data page.
2. Accept the competition rules if Kaggle prompts you.
3. Download the competition data.
4. Extract `train_transaction.csv` and `train_identity.csv` into `data/raw/`.

Optional Kaggle CLI flow:

```bash
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

After the files are in place, run:

```bash
python -m src.cli --config src/config_files/config_ieee_cis.yaml prepare-data
```

## Additional Datasets

The preparation pipeline also supports the following datasets through dedicated example configs:

- `IEEE-CIS`: `src/config_files/config_ieee_cis.yaml`
- `BAF`: `src/config_files/config_baf.yaml`
- `PaySim`: `src/config_files/config_paysim.yaml`
- `Credit Card Fraud Detection Dataset`: `src/config_files/config_credit_card_fraud.yaml`
- `Elliptic Data Set`: `src/config_files/config_elliptic.yaml`

Expected raw files by dataset:

- `BAF`
  - default config points to `data/raw/baf/Base.csv`
  - alternative raw files such as `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, and `Variant V.csv` can be selected via `paths.raw_main_path` or `--raw-main`
- `PaySim`
  - set `paths.raw_main_path` to the PaySim transaction CSV
- `Credit Card Fraud Detection Dataset`
  - set `paths.raw_main_path` to `creditcard.csv`
- `Elliptic Data Set`
  - set `paths.raw_main_path` to `elliptic_txs_features.csv`
  - set `paths.raw_aux_path` to `elliptic_txs_classes.csv`
  - `paths.raw_edge_path` can point to `elliptic_txs_edgelist.csv`

Example preparation commands:

```bash
python -m src.cli --config src/config_files/config_baf.yaml prepare-data
python -m src.cli --config src/config_files/config_paysim.yaml prepare-data
python -m src.cli --config src/config_files/config_credit_card_fraud.yaml prepare-data
python -m src.cli --config src/config_files/config_elliptic.yaml prepare-data
```

Example BAF variant override:

```bash
python -m src.cli --config src/config_files/config_baf.yaml prepare-data --raw-main "data/raw/baf/Variant I.csv"
```

Notes:

- IEEE-CIS keeps the `card4` / `card6` client mapping.
- PaySim uses transaction `type` as the client split source.
- BAF, Credit Card Fraud Detection, and Elliptic use deterministic quantile-based pseudo-clients.
- The current Elliptic pipeline uses the node-feature table plus class labels. The edge list is retained in config for provenance, but it is not modeled directly in Stage 1 or Stage 2.

## Data Contract

Raw input:

- dataset-specific raw table(s), normalized into one prepared tabular training corpus

Prepared output:

- one aligned train CSV and one aligned test CSV per client under `<prepared_root>/clients/`
- shared preprocessing artifacts under the configured `prepared_root`
- two client metadata overview figures under `<prepared_root>/client_rows_overview.png` and `<prepared_root>/client_fraud_overview.png`

Stage 1 output:

- synthetic pools for the enabled generator baselines, including `collafuse`, `ctgan`, `tabddpm`, `local_only_ddpm`, and `centralized_ddpm`
- CollaFuse checkpoint
- RQ1 artifacts: MMD summaries, embedding CSV, plots, training history, and CollaFuse training loss / denoising-accuracy curves

Stage 2 output:

- raw and aggregated classifier metrics
- paper-ready plots for `f1`, `precision`, `recall`, `roc_auc`, and `average_precision`

## Baselines

The default baseline set is controlled through `baselines.enabled_sources` in `src/config_files/config_collafuse.yaml`.

The shared config currently enables:

- `real_only_unweighted`
- `real_only_weighted`
- `random_oversampling`
- `smote`
- `adasyn`
- `collafuse`
- `ctgan`
- `tabddpm`
- `local_only_ddpm`
- `centralized_ddpm`

## Plot Styling

Stage 1 and Stage 2 plots use `SciencePlots` automatically when it is installed in the active environment.

Install it in the project `.venv` with:

```bash
.venv/bin/pip install SciencePlots
```

## Configuration

`src/config_files/config_collafuse.yaml` is now the shared base config for the common Stage 1 / Stage 2 settings.

Each dataset has a dedicated runtime config under `src/config_files/` that extends that base and only overrides dataset-specific details such as:

- data paths and preprocessing thresholds
- the dataset-specific client split strategy

The shared base carries the common:

- CollaFuse hyperparameters
- sampling controls
- augmentation ratios
- baseline source selection and generator settings
- evaluation seeds and plotting settings
- sklearn classifier suite

## Default Stage 2 Models

- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

The suite is config-driven and intentionally stays sklearn-only.

## Tests

Run:

```bash
python -m unittest discover -s tests
```
