import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Define features and target
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Define a function to predict churn from user input using Streamlit
def predict_churn_from_input():
    st.title("Churn Prediction App")

    st.write(f"Model Accuracy: {accuracy:.2f}")

    credit_score = st.number_input("Enter the customer's credit score:", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Enter the customer's geography:", ('France', 'Germany', 'Spain'))
    gender = st.selectbox("Enter the customer's gender:", ('Male', 'Female'))
    age = st.number_input("Enter the customer's age:", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Enter the customer's tenure (in years):", min_value=0, max_value=10, value=3)
    balance = st.number_input("Enter the customer's account balance:", min_value=0.0, value=10000.0)
    num_of_products = st.number_input("Enter the number of bank products used by the customer:", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Does the customer have a credit card?", (1, 0))
    is_active_member = st.selectbox("Is the customer an active member?", (1, 0))
    estimated_salary = st.number_input("Enter the customer's estimated salary:", min_value=0.0, value=50000.0)

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Convert input data to the same format as training data
    input_data = pd.get_dummies(input_data, columns=['Geography', 'Gender'], drop_first=True)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_features = sc.transform(input_data)

    # Predict churn
    churn_prediction = model.predict(input_features)
    churn_prediction_human_readable = np.where(churn_prediction == 1, 'Churn', 'No Churn')

    st.write(f"The predicted churn status for the customer is: {churn_prediction_human_readable[0]}")

# Run the Streamlit app
if __name__ == '__main__':
    predict_churn_from_input()
