import joblib
import numpy as np

# Load model yang sudah disimpan
model = joblib.load("slr_model.pkl")


def predict_purchases(target_clicks: int) -> dict:
    X = np.array([[target_clicks]])
    predicted = model.predict(X)[0]
    predicted = max(0, int(round(predicted)))

    return {
        "target_clicks": target_clicks,
        "estimated_purchases": predicted,
        "conversion_rate": f"{(predicted / target_clicks * 100):.1f}%",
    }


# Test
if __name__ == "__main__":
    result = predict_purchases(500)
    print(result)
