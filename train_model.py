import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/salary_data.csv")

# 🔹 Identify features & target
target_column = "salary"   # Change if your salary column has a different name
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔹 Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# 🔹 Handle missing values
X = X.fillna(X.median())
y = y.fillna(y.median())

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("✅ Model trained on features:", list(X.columns))

# 🔹 Save model
pickle.dump(model, open("models/trained_model.pkl", "wb"))
print("✅ Model saved as trained_model.pkl")
