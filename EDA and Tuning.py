import pandas as pd
import numpy as np
import pickle
from ridge_model import RidgeScratch   # ✅ CRITICAL IMPORT

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("nse_all_stock_data (1).csv")
data = data[["Date", "RELIANCE"]].dropna()

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data["log_price"] = np.log(data["RELIANCE"])

# =========================
# LAG FEATURES
# =========================
def make_lag_features(series, window):
    df = pd.DataFrame()
    for i in range(1, window + 1):
        df[f"lag_{i}"] = series.shift(i)
    df["target"] = series.values
    return df.dropna().reset_index(drop=True)

# =========================
# BEST PARAMETERS
# =========================
BEST_WINDOW = 5
BEST_LAMBDA = 0.1

feat = make_lag_features(data["log_price"], BEST_WINDOW)

split = int(0.8 * len(feat))
train = feat[:split]

X_train = train.drop("target", axis=1).values
y_train = train["target"].values

# =========================
# FEATURE SCALING
# =========================
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std

# =========================
# TRAIN MODEL
# =========================
model = RidgeScratch(lam=BEST_LAMBDA)
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
with open("ridge_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "mean": mean,
        "std": std,
        "window": BEST_WINDOW
    }, f)

print("✅ Model trained and saved as ridge_model.pkl")