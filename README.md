# Churn Prediction Model

## Overview

This repository contains a machine learning solution for predicting customer churn based on a logistic regression model. Customer churn refers to when customers stop using a company's products or services, which can be costly for businesses. This model helps identify customers who are likely to churn, enabling proactive retention strategies.

## Project Description

The churn prediction model analyzes various employee/customer features to predict the likelihood of them leaving. The model has been trained on historical data and can be used to make predictions on new data. This solution is particularly useful for HR departments, subscription-based businesses, telecom companies, or any organization looking to reduce customer attrition.

## Repository Structure

- `.gitignore` - Specifies files to be ignored by Git version control
- `Dockerfile` - Contains instructions to build a Docker container, enabling easy deployment
- `README.md` - Project documentation and instructions
- `app.py` - Main application file that implements the churn prediction functionality
- `logistic_regression_model.pkl` - Serialized logistic regression model trained on historical data
- `requirements.txt` - Lists all Python dependencies required for the project
- `streamlit_app.py` - Interactive web interface built with Streamlit for easy model interaction
- `test.py` - Test scripts to ensure the functionality works as expected

## Features Used in the Model

Based on the repository contents, the model uses various features to predict churn, including:
- Age
- BusinessTravel patterns
- DailyRate
- Department
- DistanceFromHome
- Education level
- EducationField
- EmployeeNumber
- EnvironmentSatisfaction
- And many other employee/customer attributes

## Technologies Used

- **Python** (97.2%) - Primary programming language
- **Docker** (2.8%) - For containerization and deployment
- **Logistic Regression** - The machine learning algorithm used for prediction
- **Streamlit** - For creating an interactive web interface

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main application:
   ```
   python app.py
   ```
4. To use the interactive Streamlit interface:
   ```
   streamlit run streamlit_app.py
   ```

## Deployment

The application can be containerized using the provided Dockerfile, making it easy to deploy in various environments.
