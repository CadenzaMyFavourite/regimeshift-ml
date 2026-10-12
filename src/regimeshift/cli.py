"""Command line interface. Run directories are append-only at study boundaries."""

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    # Set before importing Polars; cap its process-wide pool as well as model/BLAS threads.
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    parser = argparse.ArgumentParser(prog="regimeshift", description="RegimeShift-ML research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic data and run the complete study")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--dates", type=int, default=160)
    demo.add_argument("--stocks", type=int, default=12)
    demo.add_argument("--snapshots", type=int, default=24)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--execute-notebook", action="store_true")
    prep = sub.add_parser("prepare", help="Validate local CSV and write date partitions")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("data/processed/optiver"))
    prep.add_argument("--source-kind", choices=["optiver", "synthetic"], default="optiver")
    dev = sub.add_parser("develop", help="Fit, tune, calibrate and freeze a study")
    dev.add_argument("--config", type=Path, required=True)
    ev = sub.add_parser("evaluate", help="Replay final dates using verified frozen artifacts")
    ev.add_argument("--run", type=Path, required=True)
    rep = sub.add_parser("report", help="Generate figures, Markdown and research notebook")
    rep.add_argument("--run", type=Path, required=True)
    rep.add_argument("--execute-notebook", action="store_true")
    args = parser.parse_args(argv)
    from .config import Config
    from .data import prepare
    from .pipeline import develop, evaluate
    from .reporting import report
    from .synthetic import generate

    try:
        if args.command == "prepare":
            result = prepare(args.input, args.output, args.source_kind)
        elif args.command == "develop":
            result = develop(Config.load(args.config))
        elif args.command == "evaluate":
            result = evaluate(args.run)
        elif args.command == "report":
            result = report(args.run, args.execute_notebook)
        else:
            run = args.output.resolve()
            # Inputs stay outside the run directory, which develop creates atomically by stage.
            inputs = run.with_name(run.name + "-inputs")
            if run.exists() or inputs.exists():
                raise FileExistsError("Demo output or its inputs exist; choose a new --output")
            raw = generate(inputs / "synthetic.csv", args.dates, args.stocks,
                           args.snapshots, args.seed)
            dataset = prepare(raw, inputs / "processed", "synthetic")
            config = Config(dataset=str(dataset), run_dir=str(run), seed=args.seed,
                            lgbm_estimators=150, lgbm_min_child_samples=30)
            develop(config)
            evaluate(run)
            result = report(run, args.execute_notebook)
        print(f"Completed: {result}")
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
