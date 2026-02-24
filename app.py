import streamlit as st
import matplotlib.pyplot as plt
from model_utils import predict_next_price

st.set_page_config(
    page_title="Reliance Stock Predictor",
    layout="centered"
)

st.title("📈 Reliance Stock Price Prediction")
st.write(
    "Ridge Regression (from scratch) with lag-based time series features"
)

WINDOW = 5
prices = []

st.subheader("Enter last 5 closing prices (₹)")

for i in range(WINDOW):
    p = st.number_input(
        f"Day -{WINDOW - i}",
        min_value=1.0,
        step=1.0,
        key=i
    )
    prices.append(p)

if st.button("Predict Next Day Price"):
    try:
        prediction = predict_next_price(prices)

        # ---- Result box ----
        st.success(f"📊 Predicted Next Day Price: ₹ {prediction:.2f}")

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(8, 4))

        x_actual = list(range(-WINDOW, 0))
        y_actual = prices

        x_pred = [0]
        y_pred = [prediction]

        ax.plot(x_actual, y_actual, marker="o", label="Past Prices")
        ax.plot(x_pred, y_pred, marker="o", color="red", label="Predicted Price")

        ax.set_xlabel("Day")
        ax.set_ylabel("Price (₹)")
        ax.set_title("Price Trend and Prediction")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(str(e))