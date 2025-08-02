# salary-prediction
This project predicts salaries using machine learning. It has two parts: 🔹 Job-based prediction using an ensemble model 🔹 Experience-based prediction using linear regression  Built with Python and Streamlit, it provides a GUI showing salaries in INR. 

A **Machine Learning web application** built with **Python, scikit-learn, and Streamlit** that predicts salaries based on **Age, Gender, Education Level, Job Title, and Years of Experience**.

Features
✅ Train and save a salary prediction model using a CSV dataset  
✅ Predict salary with a clean web interface  
✅ Uses **RandomForest Regressor** for accurate predictions  
✅ Automatically encodes categorical features  
✅ Ready to deploy on **Streamlit Cloud (Free)**  

---

📂 Project Structure
salary-prediction/
│-- app.py # Streamlit web app
│-- train_model.py # Script to train the ML model
│-- models/
│ ├── trained_model.pkl # Saved ML model
│-- data/
│ └── salary_data.csv # Salary dataset
| └──Salary_data1.csv
│-- requirements.txt # Project dependencies
│-- README.md # Documentation

Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/Kushagrakushwaha/salary-prediction.git
cd salary-prediction

2️⃣ Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

📊 Train the Model
Place your dataset inside the data/ folder with columns:
Age, Gender, Education Level, Job Title, Years of Experience, Salary
then run: python train_model.py

🌐 Run the App: streamlit run app.py


## 📸 App Preview

![Salary Prediction Screenshot](https://raw.githubusercontent.com/kushagrakushwaha/salary-prediction/main/screenshot.png)
![Salary prediction screenshot](https://github.com/Kushagrakushwaha/salary-prediction/blob/main/Screenshot.png)
