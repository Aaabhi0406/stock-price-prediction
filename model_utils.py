import pickle
import numpy as np
from ridge_model import RidgeScratch   # 👈 REQUIRED

with open("ridge_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
mean = bundle["mean"]
std = bundle["std"]
WINDOW = bundle["window"]

def predict_next_price(last_prices):
    if len(last_prices) != WINDOW:
        raise ValueError(f"Expected {WINDOW} prices")

    log_prices = np.log(last_prices)
    X = (log_prices - mean) / std
    pred_log = model.predict(X.reshape(1, -1))[0]
    return float(np.exp(pred_log))