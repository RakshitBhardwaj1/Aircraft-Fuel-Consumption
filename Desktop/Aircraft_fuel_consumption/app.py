from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import streamlit as st

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Define input data model using Pydantic for type validation
class FuelConsumptionInput(BaseModel):
    Flight_Distance: float
    Number_of_Passengers: int
    Flight_Duration: float
    Aircraft_Type: str

@app.post("/predict")
async def predict_fuel(data: FuelConsumptionInput):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Flight_Distance': [data.Flight_Distance],
        'Number_of_Passengers': [data.Number_of_Passengers],
        'Flight_Duration': [data.Flight_Duration],
        'Aircraft_Type_Type1': [1 if data.Aircraft_Type == 'Type1' else 0],
        'Aircraft_Type_Type2': [1 if data.Aircraft_Type == 'Type2' else 0],
        'Aircraft_Type_Type3': [1 if data.Aircraft_Type == 'Type3' else 0]
    })

    # Make prediction
    try:
        fuel_consumption = model.predict(input_data)
        return {"Fuel_Consumption": fuel_consumption[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

st.title("Aircraft Fuel Consumption Predictor")
st.write("Enter flight details to predict fuel consumption:")

# Input fields
flight_distance = st.number_input("Flight Distance (km)", min_value=0.0)
num_passengers = st.number_input("Number of Passengers", min_value=1, step=1)
flight_duration = st.number_input("Flight Duration (hours)", min_value=0.0)
aircraft_type = st.selectbox("Aircraft Type", ["Type1", "Type2", "Type3"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Flight_Distance': [flight_distance],
        'Number_of_Passengers': [num_passengers],
        'Flight_Duration': [flight_duration],
        'Aircraft_Type_Type1': [1 if aircraft_type == 'Type1' else 0],
        'Aircraft_Type_Type2': [1 if aircraft_type == 'Type2' else 0],
        'Aircraft_Type_Type3': [1 if aircraft_type == 'Type3' else 0]
    })
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Fuel Consumption: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
