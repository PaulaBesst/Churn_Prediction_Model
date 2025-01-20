from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize LabelEncoder and MinMaxScaler
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 5))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Preprocess the input data
    # Label Encoding
    le_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in le_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Feature Scaling
    scale_columns = [col for col in df.columns if col != 'Attrition']
    for col in scale_columns:
        df[[col]] = scaler.fit_transform(df[[col]])

    # Make sure all expected columns are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match the model's expected input
    df = df[expected_columns]

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    return jsonify({
        'churn_prediction': int(prediction[0]),
        'churn_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
