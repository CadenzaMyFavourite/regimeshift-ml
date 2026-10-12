# RegimeShift-ML

**Does regime-aware weighting improve future forecasts over fixed models and adaptation without regimes?**

RegimeShift-ML is a local, reproducible research framework for that question. It compares frozen Ridge and LightGBM experts, a fixed blend, global adaptive weighting, and regime-aware adaptive weighting using chronological, delayed-label replay. It includes empirical prediction intervals, paired date-block uncertainty, an executable notebook, and tests for look-ahead leakage.

**Status:** the software and synthetic study are implemented. Real Optiver experiments are pending a locally supplied dataset with permitted access and use. Synthetic results below are software-validation results, not financial evidence.

Read the [detailed research and engineering guide](docs/PROJECT_GUIDE.md) for the project's purpose, architecture, full process, worked examples, technologies, and development barriers. An [editable Word version](docs/RegimeShift_ML_Project_Guide.docx) is also included. The [reviewed synthetic aggregate record](docs/synthetic-results.json) preserves the initial measured results without distributing row-level data or fitted objects. Generated run directories remain local and are not bundled with the source package.

## Quick start

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). From this directory:

```powershell
uv sync --locked --group notebook
uv run --group notebook regimeshift demo --execute-notebook
```

The default demo generates 46,080 rows: 160 dates, 12 synthetic stocks, 24 snapshots per stock/date. The output is `runs/demo/report/research.ipynb`, with an executed HTML companion, Markdown report, and five PNG figures. Open it directly or run:

```powershell
uv run --group notebook jupyter lab runs/demo/report/research.ipynb
```

Output directories are never silently overwritten. For another run, use `--output runs/demo-2` or another unused name.

## Research design

### Hypotheses and controls

1. Adaptive selection among frozen experts may improve accuracy over a static choice.
2. Conditioning recent expert skill on a causal market regime may improve on global adaptive weighting.
3. Updating interval widths with past residuals may improve observed coverage, potentially at the cost of wider intervals.

| Method | What changes after training? | Role |
|---|---|---|
| Zero | Nothing | No-signal baseline |
| Ridge | Nothing | Linear expert |
| LightGBM | Nothing | Nonlinear expert, absolute-error objective |
| Fixed blend | Nothing | Convex blend selected on tuning dates |
| Global adaptive | Expert weights, using previously revealed daily errors | Removes regime conditioning |
| Regime adaptive | Expert weights, using global and regime-specific past errors | Research treatment |

The strongest static reference, including the zero baseline, is selected on **tuning** performance. The reported final comparisons are regime adaptation minus this reference and minus global adaptation. Lower MAE is better; negative differences favor regime adaptation. A positive outcome is not required.

### Chronology

Sorted whole dates are split 60%/10%/10%/20% into training, tuning, calibration and final evaluation; boundaries use floor at 60%, 70%, and 80%. At least 50 dates are required. Every stock/time observation of a date stays in one block.

- Only training dates fit imputers, scalers, stock encodings, Ridge, LightGBM and K-means.
- Tuning selects expert settings, the fixed blend, and adaptive settings. Tuning scores are selection scores, not unbiased performance estimates.
- Each adaptation candidate starts from equal weights with no in-sample training residuals. After selection, tuning is replayed to reconstruct the selected policy's state.
- Calibration produces forward predictions with frozen choices and collects their residuals.
- Final evaluation carries the state forward. At the start of a date, the previous observed date's labels are revealed; updates precede new predictions. No same-day label is used, even though the Optiver target horizon is 60 seconds.
- Predictions are persisted before that date's labels enter the outcome queue. The final day's labels are scored but never used for a further prediction.

This is **prequential evaluation**: earlier test outcomes can affect later test weights through the already specified algorithm. They cannot change expert parameters, selected settings, preprocessing or regime clusters. Next-observed-date reveal timing is an offline assumption, not proof of a live feed's availability. Gaps in anonymized date IDs are not interpreted as a known number of elapsed trading days.

Within-day 10/30/60-second WAP lags use exact time joins. Six-observation trailing statistics are named as observation windows and reset across stock/date boundaries. They include the currently observable row, never centered or future rows. Dates, row IDs, time IDs and targets are not model features. Missing targets remain in observable feature history and are excluded from all supervised losses; no target is replaced by zero. Whole-date splits keep these short-horizon targets within their daily auction sessions.

### Regimes and adaptation

Three K-means clusters are fitted on standardized training daily summaries: median relative spread, median absolute book imbalance, median log liquidity, and standard deviation of observed 10-second WAP changes. Training-fitted imputation handles missing summaries. Today's regime is yesterday's completed summary assigned to these frozen clusters. Cluster IDs are descriptive, not established economic regimes or an intraday alarm system.

For each expert, the revealed day's MAE is divided by the training period's mean absolute target. With decay `rho = 2**(-1 / half_life)`, loss sums and effective date counts decay once per observed trading date. The global accumulator receives every valid day's losses; only the regime recorded for that day's predictions receives its regime update. Empty-label dates decay histories without contributing a fake loss. All regimes decay, including those not recently seen.

Let `L_global` and `L_regime` be their discounted means and `n_regime` the discounted date count:

```text
a = n_regime / (n_regime + 5)
L = a * L_regime + (1 - a) * L_global
weights = softmax(-L / temperature)
prediction = weights[0] * Ridge + weights[1] * LightGBM
```

An unseen regime uses the global estimate; no revealed history gives equal weights. Half-life `{5, 10, 20}` and temperature `{0.1, 0.5, 1.0}` are selected separately for global and regime policies by the MAE of actual blended tuning predictions. The fixed blend searches Ridge weights `0, .05, ..., 1`. Ridge searches alpha `{1, 10, 100}`; LightGBM searches `{15, 31}` leaves with the other settings fixed in YAML. No online-learning regret guarantee is claimed.

### Uncertainty and statistical comparison

Each method has symmetric nominal 90% prediction intervals, using the empirical 90th percentile of absolute past residuals (`higher` quantile rule). Fixed intervals use all calibration dates; rolling intervals use the latest 20 revealed dates, counting dates with missing labels in window expiration. Empty buffers yield unavailable intervals, not zero-width intervals. Missing intervals are excluded from coverage denominators and recorded explicitly in the diagnostic counts.

MAE aggregates absolute-error sums over shared finite-target rows. Date/regime tables retain both row and date support. Paired 95% uncertainty intervals resample contiguous five-date blocks 2,000 times, preserving row-count weighting and pairing. They describe uncertainty conditional on this realized replay path; they do not refit models or recompute adaptation for each resample. They are descriptive intervals for two prespecified comparisons, not a multiple-testing-adjusted discovery claim. Dependent observations and distribution shifts do not inherit nominal conformal coverage guarantees.

## Run on locally supplied Optiver data

Obtain the file through the [competition data page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data) using your own Kaggle account and verify permitted use under the [official rules](https://www.kaggle.com/competitions/optiver-trading-at-the-close/rules). The rules specify competition-only use and restrict redistribution. This project neither authenticates to Kaggle nor accepts terms for you; it does not bundle competition rows. If the intended study is outside permitted use, keep real-data execution pending rather than assuming a local copy resolves the restriction.

```powershell
uv run regimeshift prepare --input "C:\path\to\train.csv" --output data/processed/optiver
uv run regimeshift develop --config configs/optiver.yaml
uv run regimeshift evaluate --run runs/optiver-v1
uv run --group notebook regimeshift report --run runs/optiver-v1 --execute-notebook
```

Paths in YAML are relative to the directory where you invoke the command, then resolved absolutely. Run from the project root. The importer requires the usual Optiver columns: stock/date/auction-time keys, imbalance size/direction, matched size, reference/far/near/bid/ask prices, bid/ask sizes, WAP, and target. Extra columns are ignored. It validates key uniqueness, ten-second auction timestamps, numeric finiteness, price/size constraints and missingness, then sorts and writes date-partitioned Parquet. Actual row/stock/date counts are audited rather than assumed.

Imported data, row-level predictions, labels, fitted models, and notebooks generated from real runs remain local under ignored `data/` and `runs/`. Only deliberately reviewed and permitted aggregate findings should be shared. A synthetic source can be explicitly imported with `--source-kind synthetic`.

## Artifacts and interfaces

`prepare`, `develop`, `evaluate`, `report`, and `demo` are the public CLI. Use `regimeshift COMMAND --help` for arguments. Python modules separate data/features, frozen experts/regimes, adaptation/replay, evaluation, and reporting.

Each development run saves its resolved configuration, split manifest, dataset fingerprint/audit, environment and code fingerprints, dependency-lock fingerprint, candidate scores, centroids, models, calibration checkpoint, predictions and reveal-event logs. `frozen.json` hashes the artifacts consumed by evaluation. Changed artifacts, partitions, scientific code, Python versions or scientific-library versions are rejected before a new final replay. Evaluation saves row-level predictions, daily/aggregate/regime MAE, interval diagnostics, timing, paired comparisons, and final replay state.

The notebook reads these artifacts; it never fits a second pipeline. `notebooks/research.ipynb` is a portable unexecuted companion template; the run-specific copy retains executed outputs. Set `REGIMESHIFT_RUN` to select a different completed run, or open its generated notebook. Tables and plots recompute their presentation from saved metrics.

`models.joblib` and checkpoints are trusted-local Python artifacts; do not load downloaded joblib files. A final evaluation directory is never overwritten, including after interruption. Use a new run directory after a failure. Reusing the same final dates after examining results is an exploratory follow-up, even under a new run name; hashing protects artifact integrity, not researcher decisions across studies.

## Synthetic validation results

Default demo, seed 42: 96 training dates, 16 tuning dates, 16 calibration dates and 32 final dates; 9,169 finite-target final observations. The simulator rotates three observable contexts every 12 dates, changes the target response function by context, adds noise/missing labels, and introduces a modest late scale shift. Synthetic labels are not reconstructed future financial returns.

| Method | Final MAE, synthetic units |
|---|---:|
| LightGBM | 0.724831 |
| Fixed blend | 0.724831 |
| Regime adaptive | 0.735236 |
| Global adaptive | 0.736038 |
| Ridge | 1.371070 |
| Zero | 2.141152 |

The tuning-selected fixed blend placed all weight on LightGBM. Regime weighting's MAE difference against LightGBM was **+0.010405**, with 95% block interval **[-0.000147, +0.018356]**. Against global weighting it was **-0.000802**, interval **[-0.003889, +0.003474]**. Both include zero. This run provides no convincing evidence that regime conditioning improves accuracy and illustrates the cost of blending a weaker expert. These observations apply only to this synthetic experiment.

Reproduce the exact demo defaults through `demo`, and inspect `runs/demo/report/REPORT.md` and its notebook for coverage, widths, weights, trajectories and limitations. No Optiver or trading-performance result is claimed.

## Checks and limitations

```powershell
uv run --group notebook pytest -q
uv run ruff check .
uv build
```

Tests perturb current/future targets and future features, check lag boundaries and reveal attribution, exercise missing labels and unseen stocks, verify hand-calculated metrics and intervals, reject tampering, compare resumed replay, run a complete small study, and execute the notebook with five required embedded charts.

CPU training defaults to four model/BLAS threads, and the CLI caps Polars at four unless `POLARS_MAX_THREADS` is already set. Import and training materialize selected frames and require memory proportional to dataset size; this is not an out-of-core trainer. The implementation was exercised on synthetic data on Windows/Python 3.11, not on the full competition file. The available 32 GiB machine is the intended initial target; full-file memory/runtime still require measurement.

Prior-day regimes can miss fast changes. Frozen experts cannot learn new relationships. K-means occupancy may be sparse. A single chronology and selected hyperparameters limit generalization. Prediction intervals are empirical. Forecast-target MAE is not trading P&L, Sharpe, CAGR, or evidence of executable alpha. Deep temporal models, retraining, multiple datasets and longer replicated studies are future research directions.
