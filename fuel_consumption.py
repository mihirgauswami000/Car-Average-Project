import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and train model
@st.cache_data
def load_and_train():
    data = pd.read_csv("clean_fuel.csv")

    # Label encoders
    class_encoder = LabelEncoder()
    transmission_encoder = LabelEncoder()
    fuel_encoder = LabelEncoder()

    data['VEHICLE CLASS'] = class_encoder.fit_transform(data['VEHICLE CLASS'])
    data['TRANSMISSION'] = transmission_encoder.fit_transform(data['TRANSMISSION'])
    data['FUEL'] = fuel_encoder.fit_transform(data['FUEL'])

    # Features & Target
    X = data[["VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "EMISSIONS"]]
    y = data[["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model, class_encoder, transmission_encoder, fuel_encoder

# Load everything
model, class_enc, trans_enc, fuel_enc = load_and_train()

# --- UI ---
st.set_page_config(page_title="Fuel Consumption Predictor", page_icon="⛽")
st.title("🚗 Vehicle Fuel Consumption Predictor")
st.caption("Defaults set for: **Maruti Suzuki Dzire (Petrol)**")

# --- Input Section ---

# Vehicle Class
vehicle_class_options = list(class_enc.classes_)
vehicle_class = st.selectbox(
    "Vehicle Class",
    options=vehicle_class_options,
    index=vehicle_class_options.index("COMPACT")  # Dzire default
)

# Engine Size
engine_size = st.slider("Engine Size (Liters)", min_value=0.8, max_value=6.0, value=1.2, step=0.1)

# Cylinders
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, value=4, step=1)

# Transmission Dropdown (with full labels)
transmission_labels = {
    "A": "A - Automatic",
    "AM": "AM - Automated Manual",
    "AS": "AS - Automatic Select Shift",
    "AV": "AV - Continuously Variable (CVT)",
    "M": "M - Manual"
}
trans_display = [transmission_labels.get(code, code) for code in trans_enc.classes_]
transmission_choice = st.selectbox(
    "Transmission Type",
    options=trans_display,
    index=trans_display.index("M - Manual")  # Dzire default
)
transmission = transmission_choice.split(" - ")[0]

# Fuel Type Dropdown
fuel_labels = {
    "D": "D - Diesel",
    "E": "E - Ethanol",
    "N": "N - Natural Gas",
    "X": "X - Unknown",
    "Z": "Z - Petrol"
}
fuel_display = [fuel_labels.get(code, code) for code in fuel_enc.classes_]
fuel_choice = st.selectbox(
    "Fuel Type",
    options=fuel_display,
    index=fuel_display.index("Z - Petrol")  # Dzire default
)
fuel = fuel_choice.split(" - ")[0]

# Emissions
emissions = st.slider("CO₂ Emissions (g/km)", min_value=100, max_value=500, value=120, step=10)

# --- Predict Button ---
if st.button("🔍 Predict Fuel Consumption"):
    try:
        input_df = pd.DataFrame([{
            "VEHICLE CLASS": class_enc.transform([vehicle_class])[0],
            "ENGINE SIZE": engine_size,
            "CYLINDERS": cylinders,
            "TRANSMISSION": trans_enc.transform([transmission])[0],
            "FUEL": fuel_enc.transform([fuel])[0],
            "EMISSIONS": emissions
        }])

        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction Complete")
        st.metric("City Fuel Consumption", f"{prediction[0]:.2f} L/100 km")
        st.metric("Highway Fuel Consumption", f"{prediction[1]:.2f} L/100 km")
        st.metric("Combined Fuel Consumption", f"{prediction[2]:.2f} L/100 km")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
