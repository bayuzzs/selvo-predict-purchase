import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ── 1. Generate data sintetis ──────────────────────────────────────
np.random.seed(42)
n = 500

clicks = np.random.randint(50, 5000, n)
purchases = (
    (clicks * 0.08 + np.random.normal(0, clicks * 0.02, n)).clip(min=0).astype(int)
)

df = pd.DataFrame({"clicks": clicks, "purchases": purchases})

# ── 2. Split data ──────────────────────────────────────────────────
X = df[["clicks"]]
y = df["purchases"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Training ────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print(f"β₀ (intercept) : {model.intercept_:.4f}")
print(f"β₁ (slope)     : {model.coef_[0]:.4f}")

# ── 4. Evaluasi ────────────────────────────────────────────────────
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nEvaluasi Model:")
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")

if r2 >= 0.70:
    joblib.dump(model, "slr_model.pkl")
    print("\nModel disimpan sebagai slr_model.pkl")
else:
    print("\nModel belum memenuhi threshold R² ≥ 0.70, tidak disimpan.")
