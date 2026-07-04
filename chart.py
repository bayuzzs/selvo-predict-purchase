import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data dari train.py ────────────────────────────────────────
df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_predictions.csv")

# ── Regression line (dari seluruh range data) ─────────────────────
x_line = np.linspace(df["clicks"].min(), df["clicks"].max(), 300)
# β₀ dan β₁ dari test_predictions (pakai OLS fit yang sama)
from numpy.polynomial import polynomial as P
b1, b0 = np.polyfit(df["clicks"], df["purchases"], 1)
y_line = b0 + b1 * x_line

# ── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    df["clicks"], df["purchases"],
    alpha=0.4, s=18, color="#3B82F6", label="Campaign data (n=500)"
)
ax.plot(
    x_line, y_line,
    color="#EF4444", linewidth=1.8, label=f"Regression line (β₁={b1:.4f})"
)

ax.set_xlabel("Number of Clicks", fontsize=11)
ax.set_ylabel("Number of Purchases", fontsize=11)
ax.set_title("Clicks vs. Purchases — Synthetic Campaign Data", fontsize=12, pad=12)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("figure1_scatter.png", dpi=200)
plt.close()
print("Saved: figure1_scatter.png")
