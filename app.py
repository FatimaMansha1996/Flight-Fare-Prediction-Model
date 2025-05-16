from flask import Flask, request, jsonify
from flask_cors import CORS  # 🟢 Import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # 🟢 Enable CORS for all routes

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define the list of columns used during training
model_columns = [
    'stops', 'class', 'duration', 'days_left', 'airline_AirAsia',
    'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
    'airline_SpiceJet', 'airline_Vistara', 'source_Bangalore',
    'source_Chennai', 'source_Delhi', 'source_Hyderabad', 'source_Kolkata',
    'source_Mumbai', 'destination_Bangalore', 'destination_Chennai',
    'destination_Delhi', 'destination_Hyderabad', 'destination_Kolkata',
    'destination_Mumbai', 'arrival_Afternoon', 'arrival_Early_Morning',
    'arrival_Evening', 'arrival_Late_Night', 'arrival_Morning',
    'arrival_Night', 'departure_Afternoon', 'departure_Early_Morning',
    'departure_Evening', 'departure_Late_Night', 'departure_Morning',
    'departure_Night'
]

@app.route('/')
def index():
    return "✅ Flight Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([data])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)