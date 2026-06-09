from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model sekali saat server start
model = joblib.load("slr_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "clicks" not in data:
        return jsonify({"error": "Parameter 'clicks' diperlukan"}), 400

    target_clicks = int(data["clicks"])

    if target_clicks <= 0:
        return jsonify({"error": "Jumlah klik harus lebih dari 0"}), 400

    predicted = model.predict([[target_clicks]])[0]
    predicted = max(0, int(round(predicted)))

    return jsonify(
        {
            "target_clicks": target_clicks,
            "estimated_purchases": predicted,
            "conversion_rate": f"{(predicted / target_clicks * 100):.1f}%",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "slr_model.pkl"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
