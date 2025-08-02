import pickle
import pandas as pd

# Load model
model = pickle.load(open("models/trained_model.pkl", "rb"))

# Example input (Change keys as per your dataset)
input_data = {
    "Experience": 5,
    "Education": "Master",
    "City": "Delhi"
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Handle categorical encoding (match train preprocessing)
# ⚠️ Replace with the same encoding logic as in training
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category").cat.codes

# Predict
predicted_salary = model.predict(df)[0]
print(f"Predicted Salary: {predicted_salary}")
