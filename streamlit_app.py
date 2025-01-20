import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the trained model
model_path = "logistic_regression_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Initialize preprocessing tools
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 5))

# Streamlit UI setup
st.title("Churn Prediction App")
st.write("Enter employee's data below to predict the likelihood of churn:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
daily_rate = st.number_input("Daily Rate", value=800)
distance_from_home = st.number_input("Distance From Home", value=5)
education = st.selectbox("Education Level", options=[1, 2, 3, 4, 5])
environment_satisfaction = st.selectbox("Environment Satisfaction", options=[1, 2, 3, 4])
gender = st.selectbox("Gender", options=["Male", "Female"])
hourly_rate = st.number_input("Hourly Rate", value=50)
job_involvement = st.selectbox("Job Involvement", options=[1, 2, 3, 4])
job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
job_satisfaction = st.selectbox("Job Satisfaction", options=[1, 2, 3, 4])
marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
monthly_income = st.number_input("Monthly Income", value=5000)
monthly_rate = st.number_input("Monthly Rate", value=15000)
num_companies_worked = st.number_input("Number of Companies Worked", value=3)
overtime = st.selectbox("Overtime", options=["Yes", "No"])
percent_salary_hike = st.number_input("Percent Salary Hike", value=20)
performance_rating = st.selectbox("Performance Rating", options=[1, 2, 3, 4])
relationship_satisfaction = st.selectbox("Relationship Satisfaction", options=[1, 2, 3, 4])
stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value=3, value=1)
total_working_years = st.number_input("Total Working Years", value=10)
training_times_last_year = st.number_input("Training Times Last Year", value=3)
work_life_balance = st.selectbox("Work-Life Balance", options=[1, 2, 3, 4])
years_at_company = st.number_input("Years at Company", value=5)
years_in_current_role = st.number_input("Years in Current Role", value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", value=2)
years_with_curr_manager = st.number_input("Years with Current Manager", value=3)

# Create DataFrame for input
input_data = pd.DataFrame({
    "Age": [age],
    "DailyRate": [daily_rate],
    "DistanceFromHome": [distance_from_home],
    "Education": [education],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Gender": [gender],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked],
    "OverTime": [overtime],
    "PercentSalaryHike": [percent_salary_hike],
    "PerformanceRating": [performance_rating],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "StockOptionLevel": [stock_option_level],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training_times_last_year],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "YearsInCurrentRole": [years_in_current_role],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "YearsWithCurrManager": [years_with_curr_manager]
})

# Preprocessing
# Label Encoding
le_columns = [
    "Attrition", "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]
for col in le_columns:
    if col in input_data.columns:
        input_data[col] = le.fit_transform(input_data[col])

# One-hot encoding
input_data = pd.get_dummies(input_data, drop_first=True)

# Feature Scaling
scale_columns = [col for col in input_data.columns if col != "Attrition"]
input_data[scale_columns] = scaler.fit_transform(input_data[scale_columns])

# Ensure the input matches the model's expected features
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[expected_columns]

# Predict churn
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    st.write("Prediction:", "Churn" if prediction[0] else "No Churn")
    st.write("Churn Probability:", round(probability, 2))
