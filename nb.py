import streamlit as st
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample Hypothetical Data
data = pd.DataFrame({
    'Fever': [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    'Cough': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    'Fatigue': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'Travel_History': [1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    'COVID_Positive': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
})

# Display the data
st.title("Bayesian Network for COVID-19 Diagnosis")
st.write("Sample Data:")
st.write(data)

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('Travel_History', 'COVID_Positive'),
    ('Fever', 'COVID_Positive'),
    ('Cough', 'COVID_Positive'),
    ('Fatigue', 'COVID_Positive')
])

# Fit the model using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

# Input for symptoms
st.write("Enter symptoms to diagnose:")
fever = st.selectbox("Fever", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
cough = st.selectbox("Cough", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
fatigue = st.selectbox("Fatigue", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
travel_history = st.selectbox("Travel History", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict the probability of COVID-19
query_result = inference.query(variables=['COVID_Positive'], evidence={
    'Fever': fever,
    'Cough': cough,
    'Fatigue': fatigue,
    'Travel_History': travel_history
})

# Display the result
st.write("Probability of being COVID-19 positive:")
st.write(query_result)


