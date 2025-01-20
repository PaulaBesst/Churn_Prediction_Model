# Churn Prediction Model API

This API provides a prediction endpoint for customer churn based on a logistic regression model.

## API Endpoint

### POST /predict

Predicts the likelihood of customer churn based on input features.

#### Request Body

JSON object with customer features (match these to your model's input features):

```json
{
  "Age": 35,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 1102,
  "Department": "Sales",
  "DistanceFromHome": 1,
  "Education": 2,
  "EducationField": "Life Sciences",
  "EmployeeNumber": 1,
  "EnvironmentSatisfaction": 3,
  ...
}
