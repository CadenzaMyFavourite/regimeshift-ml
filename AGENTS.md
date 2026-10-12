# RegimeShift-ML development conventions

- Use Python 3.11 with `uv`; keep `uv.lock` updated when dependencies change.
- Preserve the predict-before-reveal boundary. Prediction APIs must not accept current labels.
- Fit preprocessing, encodings, experts and regime clusters only on training dates.
- Tests must cover chronology or statistical behavior, not just mirror implementation details.
- Keep real data, fitted artifacts, row-level outputs and real-data notebook outputs under ignored directories.
- Never turn synthetic results into financial-performance claims. Real-data study status stays pending until permitted input has actually been evaluated.
- Run `uv run --group notebook pytest -q` and `uv run ruff check .` for implementation changes. Reporting changes must retain five embedded notebook charts and be visually inspected.
- Do not refit or change settings after seeing the final period and describe the result as an untouched evaluation.
