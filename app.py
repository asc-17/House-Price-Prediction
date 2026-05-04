from pathlib import Path

import joblib
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

# Load model once and cache it
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(area: float) -> float:
    """Predict house price from area."""
    model = load_model()
    prediction = model.predict(np.array([[area]]))
    return float(prediction.flatten()[0])


# Streamlit app layout
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Predictor")
st.write("Enter the house area and get an instant price estimate using linear regression.")

with st.form("prediction_form"):
    area = st.number_input(
        "House Area (sq.m)",
        min_value=0.0,
        value=1000.0,
        step=10.0,
        help="Enter the total area of the house in square meters.",
    )
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    try:
        prediction = round(predict_price(area), 2)
        st.success(f"✅ Predicted house price: **${prediction:,.0f}**")
        st.info(f"Input: {area} sq.m → Output: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"❌ Error generating prediction: {str(e)}")

st.divider()
st.caption("💡 Model: Linear Regression | Input: House Area | Output: Predicted Price")