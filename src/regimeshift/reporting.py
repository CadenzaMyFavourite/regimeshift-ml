"""Portable Matplotlib figures, Markdown findings and an artifact-reading notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .utils import read_json

COLORS = {"zero": "#6B7280", "ridge": "#BE741B", "lightgbm": "#28699C",
          "fixed": "#847440", "global": "#A14F74", "regime": "#173E65"}
LABELS = {"zero": "Zero", "ridge": "Ridge", "lightgbm": "LightGBM",
          "fixed": "Fixed blend", "global": "Global adaptive", "regime": "Regime adaptive"}


def plot_results(run: Path) -> dict:
    root = run / "evaluation"
    summary = read_json(root / "summary.json")
    reviewed_dates = summary["test_dates"]
    date_ticks = [reviewed_dates[i] for i in np.linspace(
        0, len(reviewed_dates) - 1, min(7, len(reviewed_dates)), dtype=int)]
    prefix = "SYNTHETIC • " if summary["source_kind"] == "synthetic" else ""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "figure.facecolor": "white", "axes.titlepad": 14})
    figures = {}
    metrics = pl.read_parquet(root / "metrics.parquet").sort("mae", descending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    names, values = metrics["model"].to_list(), metrics["mae"].to_numpy()
    ax.barh([LABELS[m] for m in names], values, color=[COLORS[m] for m in names])
    ax.set_xlim(0, max(values) * 1.18)
    for i, value in enumerate(values):
        ax.text(value + max(values) * .015, i, f"{value:.4f}", va="center")
    ax.set(title=prefix + "Final-period forecast error", xlabel="Mean absolute error (target units)")
    figures["model_comparison"] = fig

    daily = pl.read_parquet(root / "daily_metrics.parquet")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    reference = summary["static_reference"]
    for model, style in [(reference, "--"), ("global", ":"), ("regime", "-")]:
        data = daily.filter(pl.col("model") == model).sort("date_id")
        ax.plot(data["date_id"], data["mae"], linestyle=style, color=COLORS[model],
                label=LABELS[model], linewidth=1.8)
    ax.set(title=prefix + "Error through the final period", xlabel="Trading date ID",
           ylabel="Daily MAE (target units)")
    ax.set_xticks(date_ticks)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    figures["daily_errors"] = fig

    import json
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    dates = [e["date_id"] for e in events]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1]})
    for model, style in [("global", "--"), ("regime", "-")]:
        axes[0].plot(dates, [e[f"{model}_ridge_weight"] for e in events], style,
                     color=COLORS[model], label=LABELS[model], linewidth=1.8)
    axes[0].set(title=prefix + "Adaptive weights and causal regime assignments",
                ylabel="Ridge weight", ylim=(0, 1))
    axes[0].legend(ncols=2, loc="upper left")
    axes[1].step(dates, [e["regime"] for e in events], where="mid", color="#6B7280")
    axes[1].set(xlabel="Trading date ID", ylabel="Prior-day regime", yticks=[0, 1, 2])
    axes[1].set_xticks(date_ticks)
    figures["weights_regimes"] = fig

    intervals = pl.read_parquet(root / "interval_metrics.parquet").filter(
        pl.col("model").is_in([reference, "global", "regime"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    models = [reference, "global", "regime"]
    for j, kind in enumerate(["fixed", "rolling"]):
        rows = [intervals.filter((pl.col("model") == m) & (pl.col("interval") == kind))
                .row(0, named=True) for m in models]
        for ax, column in zip(axes, ["coverage", "mean_width"]):
            ax.bar(np.arange(3) + (j - .5) * .35,
                   [row[column] if row[column] is not None else np.nan for row in rows], width=.35,
                   color="#28699C" if j == 0 else "#BE741B",
                   hatch="" if j == 0 else "//", label=kind.capitalize())
            ax.set_xticks(np.arange(3), [LABELS[m].replace(" ", "\n") for m in models])
    axes[0].axhline(summary["coverage_target"], color="#444444", linestyle="--",
                    label=f"Nominal {summary['coverage_target']:.0%}")
    axes[0].set(ylabel="Empirical coverage", ylim=(0, 1.05), title="Interval coverage")
    axes[1].set(ylabel="Mean interval width (target units)", title="Interval width")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[1].legend(fontsize=9, loc="lower left")
    fig.suptitle(prefix + "Uncertainty diagnostics on identical forecast rows")
    figures["intervals"] = fig

    comparisons = summary["comparisons"]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, comparison in enumerate(comparisons):
        value, low, high = [comparison[k] for k in ["mae_difference", "ci_low", "ci_high"]]
        ax.hlines(i, low, high, color="#173E65", linewidth=2)
        ax.plot(value, i, "o", color="#173E65")
        ax.text(high, i + .15, f"{value:+.4f} [{low:+.4f}, {high:+.4f}]", fontsize=9)
    ax.axvline(0, color="#6B7280", linestyle="--")
    ax.set_yticks(range(len(comparisons)), ["vs " + LABELS[c["reference"]] for c in comparisons])
    ax.set(title=prefix + "Paired MAE differences with 95% block intervals",
           xlabel="Regime adaptive − reference MAE; negative favors regime adaptation",
           ylim=(-.5, 1.7))
    ax.margins(x=.3)
    figures["paired_differences"] = fig
    return figures


def interpretation(summary):
    lines = []
    for result in summary["comparisons"]:
        direction = "lower" if result["mae_difference"] < 0 else "higher"
        interval = "excludes zero" if result["ci_low"] > 0 or result["ci_high"] < 0 else "includes zero"
        lines.append(
            f"Against {LABELS[result['reference']]}, regime adaptation has {direction} MAE "
            f"(difference {result['mae_difference']:+.4f}; 95% block interval "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], which {interval})."
        )
    if summary["source_kind"] == "synthetic":
        lines.append("These are synthetic engineering results, not evidence of financial forecasting skill.")
    return "\n\n".join(lines)


def make_notebook(run, destination, summary):
    import nbformat as nbf

    notebook = nbf.v4.new_notebook()
    # Notebook is written under run/report and uses only frozen artifacts by default.
    setup = '''import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown, display

from regimeshift.reporting import interpretation, plot_results
from regimeshift.utils import read_json

get_ipython().run_line_magic("matplotlib", "inline")
pl.Config.set_tbl_rows(24)
pl.Config.set_float_precision(4)

default_run = (Path.cwd().parent if Path.cwd().name == "report" else
               Path.cwd().parent / "runs/demo" if Path.cwd().name == "notebooks" else
               Path("runs/demo"))
RUN = Path(os.environ.get("REGIMESHIFT_RUN", str(default_run))).resolve()
summary = read_json(RUN / "evaluation/summary.json")
display(Markdown(f"**Source:** {summary['source_kind']} | **Units:** {summary['target_units']}"))
display(Markdown(interpretation(summary)))
'''
    notebook.cells = [
        nbf.v4.new_markdown_cell("# RegimeShift-ML research companion\n\n## tl;dr\n\n"
            "The summary below is computed from the selected saved run. Synthetic runs validate "
            "the software only. Change `REGIMESHIFT_RUN` to inspect another completed study.\n\n"
            "### Load saved artifacts\n\nUse the project's Python 3.11 environment. No training occurs here."),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## Context & Methods\n\nThe question is whether regimes improve adaptive blending "
            "of frozen Ridge and LightGBM experts. Compare the chosen static reference and a global adaptive "
            "control. Models use 60% of dates; tuning, calibration and final evaluation use 10%, 10%, 20%.\n\n"
            "### Key Assumptions\n\nLabels arrive on the next observed date. Today's regime uses yesterday's "
            "completed summary. Hyperparameters stay frozen during final evaluation. Bootstrap intervals "
            "resample contiguous dates on the realized prediction path; they do not rerun model fitting. "
            "Nominal prediction-interval coverage is not guaranteed under dependence or shifts."),
        nbf.v4.new_markdown_cell("## Data\n\nThe dataset manifest records the file fingerprint, missingness, "
            "and date counts. Real Optiver rows stay local. Synthetic labels are simulated response functions, "
            "not actual 60-second returns.\n\nSource terms: "
            "[Optiver competition rules](https://www.kaggle.com/competitions/optiver-trading-at-the-close/rules)."),
        nbf.v4.new_code_cell('manifest = read_json(RUN / "dataset.json")\n'
                            'splits = read_json(RUN / "splits.json")\n'
                            'display(pl.DataFrame([{ "block": k, "dates": len(v), "first": v[0], "last": v[-1]} '
                            'for k, v in splits.items()]))\n'
                            'display(pl.DataFrame([{"rows": manifest["audit"]["rows"], '
                            '"valid_targets": manifest["audit"]["valid_targets"], '
                            '"stocks": manifest["audit"]["stocks"]}]))'),
        nbf.v4.new_markdown_cell("## Results\n\nAll model comparisons use the same finite-target rows. "
            "The static reference was selected on tuning dates, before final evaluation."),
        nbf.v4.new_code_cell('display(pl.read_parquet(RUN / "evaluation/metrics.parquet").sort("mae"))\n'
                            'figures = plot_results(RUN)\nplt.close("all")\ndisplay(figures["model_comparison"])'),
        nbf.v4.new_markdown_cell("### Performance over time\n\nDaily error trajectories show whether "
                                 "the aggregate comparison hides unstable performance."),
        nbf.v4.new_code_cell('display(figures["daily_errors"])'),
        nbf.v4.new_markdown_cell("### Adaptation and regime support\n\nWeights change only after a prior "
            "date's labels are revealed. Cluster IDs are descriptive, not economic ground truth. "
            "The global control removes regime conditioning while retaining online weighting."),
        nbf.v4.new_code_cell('display(figures["weights_regimes"])\n'
                            'display(pl.read_parquet(RUN / "evaluation/regime_metrics.parquet")'
                            '.select("regime_id", "model", "count", "scored_dates", "mae"))'),
        nbf.v4.new_markdown_cell("### Calibration\n\nInterpret coverage together with interval width. "
            "A wider interval can improve coverage without improving point forecasts."),
        nbf.v4.new_code_cell('display(figures["intervals"])\n'
                            'display(pl.read_parquet(RUN / "evaluation/interval_metrics.parquet")'
                            '.select("model", "interval", "count", "coverage", "mean_width"))'),
        nbf.v4.new_markdown_cell("### Paired uncertainty\n\nDifferences are regime-aware minus reference "
            "MAE. Intervals use the configured contiguous-date moving blocks and preserve row-count weighting."),
        nbf.v4.new_code_cell('display(figures["paired_differences"])'),
        nbf.v4.new_markdown_cell("## Takeaways\n\n"
            "This is a study of selecting among frozen experts. It cannot establish gains from retraining, "
            "intraday detection, trading profits, or a novel learning algorithm. Real-data research remains "
            "pending when this notebook uses synthetic data."),
        nbf.v4.new_code_cell('display(Markdown(interpretation(summary)))'),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python 3 (RegimeShift environment)",
                                   "language": "python", "name": "python3"}
    nbf.validate(notebook)
    nbf.write(notebook, destination)


def report(run: Path, execute_notebook=False):
    run = run.resolve()
    summary = read_json(run / "evaluation/summary.json")
    if summary["status"] != "complete":
        raise ValueError("A completed evaluation is required")
    destination = run / "report"
    destination.mkdir(exist_ok=True)
    figures = plot_results(run)
    for name, fig in figures.items():
        fig.savefig(destination / f"{name}.png", dpi=160)
        plt.close(fig)
    rows = ["| Model | Final MAE | Scored rows |", "|---|---:|---:|"]
    for row in sorted(summary["metrics"], key=lambda r: r["mae"]):
        rows.append(f"| {LABELS[row['model']]} | {row['mae']:.6f} | {row['count']:,} |")
    text = "# RegimeShift-ML research results\n\n"
    text += f"**Source: {summary['source_kind'].upper()}. Units: {summary['target_units']}.**\n\n"
    text += interpretation(summary) + "\n\n" + "\n".join(rows) + "\n\n"
    text += "## Method\n\nFrozen experts; training-only features and clustering; chronological 60/10/10/20 "
    text += "date split; next-date label reveals; tuning-selected static reference. Global weighting is "
    text += "the ablation for regime conditioning. Fixed blending is the ablation for online weighting.\n\n"
    for name in figures:
        text += f"![{name.replace('_', ' ')}]({name}.png)\n\n"
    text += "## Limitations\n\n" + "\n".join("- " + x for x in summary["limitations"])
    text += "\n- Regimes describe prior-day conditions; this is not intraday shift detection.\n"
    text += "- Aggregate forecast errors are not executable trading returns.\n"
    text += "- Synthetic runs validate software behavior only; real-data experiments remain pending.\n" \
            if summary["source_kind"] == "synthetic" else ""
    text += "\n## Provenance\n\nSee `../dataset.json`, `../splits.json`, `../selection.json`, "
    text += "`../frozen.json`, and `../evaluation/summary.json`. Row-level outputs are private local artifacts.\n"
    (destination / "REPORT.md").write_text(text, encoding="utf-8")
    notebook_path = destination / "research.ipynb"
    make_notebook(run, notebook_path, summary)
    if execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter

        notebook = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                                resources={"metadata": {"path": str(destination)}})
        client.execute()
        nbformat.write(notebook, notebook_path)
        body, _ = HTMLExporter().from_notebook_node(notebook)
        (destination / "research.html").write_text(body, encoding="utf-8")
    return destination
