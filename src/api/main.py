from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path("model")
MODEL_CANDIDATES = [
    MODEL_DIR / "best_model.joblib",
    MODEL_DIR / "best_model.pkl",
]
SCALER_CANDIDATES = [
    MODEL_DIR / "scaler.joblib",
    MODEL_DIR / "scaler.pkl",
]
IMPUTER_CANDIDATES = [
    MODEL_DIR / "imputer.joblib",
    MODEL_DIR / "imputer.pkl",
]


class HeartDiseaseRequest(BaseModel):
    age: float = Field(..., ge=0)
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk: str


app = FastAPI(title="Heart Disease Inference API", version="1.0.0")
model = None
scaler = None
imputer = None


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@app.on_event("startup")
def load_model() -> None:
    global model, scaler, imputer

    model_path = _resolve_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        raise RuntimeError("Model file not found in model/. Run training first.")
    model = joblib.load(model_path)

    scaler_path = _resolve_existing_path(SCALER_CANDIDATES)
    imputer_path = _resolve_existing_path(IMPUTER_CANDIDATES)
    scaler = joblib.load(scaler_path) if scaler_path else None
    imputer = joblib.load(imputer_path) if imputer_path else None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "imputer_loaded": imputer is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseRequest) -> PredictionResponse:
    try:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        feature_frame = pd.DataFrame([payload.model_dump()])

        if scaler is not None and imputer is not None:
            transformed = imputer.transform(feature_frame)
            transformed = scaler.transform(transformed)
            prediction = int(model.predict(transformed)[0])
            probability = float(model.predict_proba(transformed)[0][1])
        else:
            prediction = int(model.predict(feature_frame)[0])
            probability = float(model.predict_proba(feature_frame)[0][1])

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk="High" if probability >= 0.5 else "Low",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
