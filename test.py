import requests

# URL of the Flask app
url = "http://127.0.0.1:5000/predict"

# Sample data to test the model
test_data = {
    "Age": 34,
    "DailyRate": 800,
    "DistanceFromHome": 5,
    "Education": 3,
    "EnvironmentSatisfaction": 4,
    "Gender": "Male",
    "HourlyRate": 50,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 4,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 3,
    "OverTime": "Yes",
    "PercentSalaryHike": 20,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 3
}

# Send POST request
response = requests.post(url, json=test_data)

# Print the response
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print(f"Failed to get a valid response. Status code: {response.status_code}")
    print("Response text:", response.text)
