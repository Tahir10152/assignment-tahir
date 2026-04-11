# Assignment Python

Overview

the project contains a simplified assignment pipeline.
It loads `train.csv`, `ideal.csv`, and `test.csv`, selects matching ideal functions,
maps test points to ideal functions, saves CSV snapshots under `db/`, and produces
PNG visualizations under `visuals/`.

Prerequisites

- Python 3.8+
- Install dependencies:
 * Numpy
 * Pandas
 * sqlalchamy
 * bokeh
 * 


```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
```

Run

From the `assignment-tahir` folder run:

```bash
python main.py
```

Continuous integration

This repository includes a GitHub Actions workflow that runs tests on push/PR.
After pushing to GitHub you should see a workflow at `.github/workflows/ci.yml`.


Outputs

- Visuals: `visuals/training_vs_ideal.png`, `visuals/mappings.png`
- Persisted CSVs: `db/training.csv`, `db/ideal.csv`, `db/mappings.csv` (if mappings exist)

Notes

- If `python` is not available in your environment, install Python or run inside a suitable container.
- The implementation is intentionally different from the original project (different selectors, mapper thresholds, and matplotlib-based visualizer).

Regards

- contacts: mtahir10152@gmail.com
