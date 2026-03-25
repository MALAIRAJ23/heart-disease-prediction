from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease models with MLflow tracking.")
    parser.add_argument("--data-path", default="data/heart_disease.csv", help="Path to training dataset CSV")
    parser.add_argument("--model-dir", default="model", help="Directory to persist model artifacts")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="heart-disease-prediction", help="MLflow experiment name")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_path=Path(args.data_path),
        model_dir=Path(args.model_dir),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    summary = train(config)
    print("Training complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
