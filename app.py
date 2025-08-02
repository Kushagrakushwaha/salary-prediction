import streamlit as st
import pandas as pd
import pickle

st.title("💰 Salary Prediction App")

# Load trained model
model = pickle.load(open("models/trained_model.pkl", "rb"))

# ✅ Define all columns the model was trained on
trained_features = ['Job Title', 'Salary Estimate', 'Job Description', 'Rating', 'Company Name',
                    'Location', 'Headquarters', 'Size', 'Founded', 'Type of ownership',
                    'Industry', 'Sector', 'Revenue', 'Competitors', 'hourly',
                    'employer_provided', 'min_salary', 'max_salary', 'avg_salary', 'company_txt',
                    'job_state', 'same_state', 'age', 'python_yn', 'R_yn', 'spark', 'aws', 'excel']

st.subheader("Enter Employee Details")

company = st.text_input("Company Name")
job_title = st.text_input("Job Title")
city = st.text_input("City")
experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
industry = st.text_input("Industry")

if st.button("Predict Salary"):
    # 🔹 Create a row with default values
    input_data = {col: 0 for col in trained_features}

    # 🔹 Fill only the selected inputs
    input_data["Company Name"] = company
    input_data["Job Title"] = job_title
    input_data["Location"] = city
    input_data["age"] = experience
    input_data["Industry"] = industry

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    salary = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")
