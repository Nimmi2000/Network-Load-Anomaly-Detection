# Network Load Anomaly Detection

This repository implements an end-to-end unsupervised anomaly detection pipeline for network device load metrics. The project combines time-series exploratory analysis with a learning-based model (Embed-LSTM style sequence model) and clustering of reconstruction/error signals (DBSCAN) for identifying anomalous device/interface behaviour.

Key points and findings (from the exploratory notebook in `scratch/Explanatory_notebook.ipynb`):

- Data source: `device_data.csv` (columns observed: `timeslot`, `hostname`, `ifname`, `max_metric_value`, `mean_metric_value`). The notebook parses `timeslot` as a datetime column.
- No missing values were found in the dataset used for the notebook, so no imputation was necessary before processing.
- Exploratory plots (time-series and boxplots) show that some routers/interfaces have large values and outliers; therefore feature scaling/normalization is required so higher magnitude series do not dominate training.
- Distribution checks (KS tests across router/interface combinations) show that different routers can follow different distributions — useful to know when deciding whether to train a single global model or per-device models.
- Auto/cross-correlation analysis indicates a strong daily (24-hour) pattern for most routers. The notebook uses this to justify a 24-step input window for time-series modeling.

Repository layout (important files)

- `src/data_preparation.py`  - data loading and preprocessing utilities
- `src/training.py`        - training script for the model
- `src/model.py`           - model definition(s)
- `src/test.py`            - inference / evaluation script that produces prediction outputs
- `src/resources/config.py`- configuration helper (paths, hyperparameters)
- `scratch/Explanatory_notebook.ipynb` - exploratory analysis and reasoning used to choose preprocessing and windowing
- `src/datasets/`          - sample datasets used during development (e.g., `device_data.csv`)
- `data/output/Example_prediction.csv` - example output of inference
- `src/saved_binary_files/best_model.h5` - a trained model checkpoint included for example/inference
- `requirements.txt`      - Python package dependencies used by the project

Quickstart (Windows PowerShell)

1) Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Prepare the data (this will load `device_data.csv` and perform preprocessing used by training):

```powershell
python src/data_preparation.py
```

4) Train the model:

```powershell
python src/training.py
```

5) Run inference / generate predictions:

```powershell
python src/test.py
```

Outputs

- Trained model artifact(s) are saved in `src/saved_binary_files/` (for example `best_model.h5`).
- Example predictions are available in `data/output/Example_prediction.csv`.

Modeling notes and recommendations

- Window size: the exploratory notebook shows a strong 24-hour seasonality for most routers; using a 24-step window for sequence input is a sensible default.
- Scaling: scale features (for example, MinMax or Standard) per-device or globally depending on whether you train per-device models. The notebook highlights that some devices have much larger ranges and will dominate training if not scaled.
- Distribution differences between routers (KS test) suggest considering per-device models when behavior is heterogeneous.
- Clustering reconstruction errors (DBSCAN) is an effective unsupervised way to separate rare/abnormal points from nominal behaviour.

Extending and improving

- Add unit tests for preprocessing and training pipelines.
- Add a small CLI or Makefile to standardize running the pipeline.
- Add evaluation metrics and visualizations for anomaly detection quality (precision/recall on labelled subsets, if available).

License & Credits

This project is distributed under the terms of the repository LICENSE file. See `LICENSE` for details.

If you need a trimmed version of this README for documentation or a short project summary, tell me what to emphasize (architecture, how to run, reproducibility) and I'll adjust it.
