# src/api.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class for a given input.

    Expects a JSON object with feature values in the 'features' field.

    Example:
    {"features": [0.2, 0.5, 1.0, 0.3]}

    Returns:
    JSON with the prediction.
    """
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
