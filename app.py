import os
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static"),
)
API_BASE_URL = os.getenv("INFERENCE_API_URL", "http://127.0.0.1:8000")


def _build_payload_from_form(form_data) -> dict:
    return {
        "age": float(form_data["age"]),
        "sex": float(form_data["sex"]),
        "cp": float(form_data["cp"]),
        "trestbps": float(form_data["trestbps"]),
        "chol": float(form_data["chol"]),
        "fbs": float(form_data["fbs"]),
        "restecg": float(form_data["restecg"]),
        "thalach": float(form_data["thalach"]),
        "exang": float(form_data["exang"]),
        "oldpeak": float(form_data["oldpeak"]),
        "slope": float(form_data["slope"]),
        "ca": float(form_data["ca"]),
        "thal": float(form_data["thal"]),
    }


def _predict_via_api(payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        payload = _build_payload_from_form(request.form)
        result = _predict_via_api(payload)

        return render_template('result.html', 
                             prediction=result["prediction"],
                             probability=round(result["probability"] * 100, 2))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Flask passthrough endpoint for backward compatibility."""
    try:
        data = request.get_json()
        result = _predict_via_api(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

