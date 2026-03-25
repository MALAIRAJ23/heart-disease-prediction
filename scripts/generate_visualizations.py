from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import create_model_comparison_chart, evaluate


if __name__ == "__main__":
    metrics = evaluate()
    create_model_comparison_chart()
    print("Evaluation artifacts generated")
    print(metrics)
