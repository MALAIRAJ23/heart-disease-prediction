# Heart Disease Prediction - Cleaned ML Engineering Structure

This repository has been safely refactored into a production-style layout while preserving training, evaluation, API serving, and Flask frontend rendering.

## Clean Project Structure

```text
project_root/
├── data/
│   └── heart_disease.csv
├── model/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── imputer.joblib
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── api/
│       └── main.py
├── frontend/
│   ├── static/
│   └── templates/
├── scripts/
│   ├── download_dataset.py
│   ├── generate_visualizations.py
│   └── model_development.py
├── archive/
│   ├── notebook.ipynb
│   └── model_legacy/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run Instructions

### 1) Flask frontend

```bash
python app.py
```

### 2) FastAPI inference API

```bash
python -m uvicorn src.api.main:app --reload
```

### 3) Model training

```bash
python scripts/model_development.py
```

### 4) Evaluation + visualization assets

```bash
python scripts/generate_visualizations.py
```

## Notes

- Flask now reads templates/static from `frontend/templates` and `frontend/static`.
- FastAPI loads model artifacts from `model/` with support for:
  - `best_model.joblib` (preferred)
  - legacy fallback `best_model.pkl` if present
- Training persists artifacts to `model/` and logs experiments to MLflow (`mlruns/`).
- Legacy/uncertain files were moved to `archive/` instead of hard-deleting for safety.
"# heart-disease-prediction" 
