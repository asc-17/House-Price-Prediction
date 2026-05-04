from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>House Price Predictor</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            color: #1f2937;
        }
        .wrapper {
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 24px;
        }
        .card {
            width: 100%;
            max-width: 520px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
            padding: 32px;
        }
        h1 {
            margin: 0 0 8px;
            font-size: 2rem;
        }
        p {
            margin: 0 0 24px;
            color: #6b7280;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
        }
        input[type="number"] {
            width: 100%;
            box-sizing: border-box;
            padding: 14px 16px;
            border: 1px solid #d1d5db;
            border-radius: 12px;
            font-size: 1rem;
            margin-bottom: 18px;
        }
        button {
            width: 100%;
            border: none;
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 1rem;
            font-weight: 700;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #1d4ed8;
        }
        .result {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .error {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background: #fef2f2;
            color: #b91c1c;
            font-weight: 700;
        }
        .hint {
            margin-top: 10px;
            font-size: 0.92rem;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="card">
            <h1>House Price Predictor</h1>
            <p>Enter the house area and get a price estimate from the trained linear regression model.</p>

            <form method="post">
                <label for="area">House area</label>
                <input
                    type="number"
                    id="area"
                    name="area"
                    min="0"
                    step="any"
                    placeholder="For example: 1200"
                    value="{{ area_value if area_value is not none else '' }}"
                    required
                >
                <button type="submit">Predict Price</button>
            </form>

            {% if prediction is not none %}
                <div class="result">Predicted house price: ${{ prediction }}</div>
            {% endif %}

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            <div class="hint">The model expects one numeric input: area.</div>
        </div>
    </div>
</body>
</html>
"""


def predict_price(area: float) -> float:
    prediction = model.predict(np.array([[area]]))
    return float(prediction[0])


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    area_value = None

    if request.method == "POST":
        area_raw = request.form.get("area", "").strip()
        area_value = area_raw

        try:
            area = float(area_raw)
            prediction = round(predict_price(area), 2)
        except ValueError:
            error = "Please enter a valid numeric area value."
        except Exception:
            error = "Unable to generate a prediction right now."

    return render_template_string(
        PAGE_TEMPLATE,
        prediction=prediction,
        error=error,
        area_value=area_value,
    )


if __name__ == "__main__":
    app.run(debug=True)