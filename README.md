# Loan-Risk-Scorer
🏦 Loan Risk Scorer

A web application that predicts the risk of loan default using applicant financial and demographic data. Built with Python and XGBoost, the app outputs both the risk class (Low/High) and the probability of default, and is deployed on Render for real-time predictions.
🔗 Live Demo: https://loan-risk-scorerrrr.onrender.com

💡 Project Overview
Financial institutions need reliable tools to assess whether loan applicants are likely to default. This project implements a full ML pipeline that:
Preprocesses raw applicant data
Trains an XGBoost model to classify loan default risk
Bundles preprocessing and model in a single pipeline
Provides a web interface for instant predictions
Outputs both risk classification and probability of default
Outputs:
✅ Low Risk
⚠️ High Risk
📊 Probability of default (0–100%)

🛠 Tech Stack
Python: Data processing & ML
Pandas & NumPy: Data manipulation
Scikit-learn: Preprocessing & pipeline
XGBoost: Classification model
Joblib: Model serialization
Streamlit: Web interface
Render: Cloud deployment

📂 Project Structure

loan-risk-scorer/
│
├── loan_default_app.py        # Streamlit web app
├── loan_pipeline_final.pkl    # Saved preprocessing + model pipeline
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version for Render
├── README.md
└── data/
    └── Loan_default.csv  # Dataset for training
    
⚙ How It Works
Data Preprocessing
Handles missing values
Encodes categorical variables
Scales numerical features
Built as a Scikit-learn pipeline
Model Training
XGBoost classifier for binary classification
Entire pipeline (preprocessing + model) saved for production
Prediction & Deployment
Users input applicant details via Streamlit UI
Model predicts both risk class and probability of default
Hosted on Render for cloud access

🔹 Features Used
Age
Income
Loan Amount
Credit History
Employment Status
Existing Loans
Loan Term

📊 Model Performance Metrics
The XGBoost model was evaluated on the test dataset (51,070 samples).
Overall Accuracy: 0.696
ROC-AUC Score: 0.732
Confusion Matrix:
Predicted \ Actual	No Default (0)	Default (1)
No Default (0)	31,747	2,117
Default (1)	13,423	3,783
Classification Report:
Class	Precision	Recall	F1-score	Support
0 (No Default)	0.94	0.70	0.80	45,170
1 (Default)	0.22	0.64	0.33	5,900
Weighted Averages:
Metric	Value
Accuracy	0.70
Precision	0.85
Recall	0.70
F1-score	0.75
ROC-AUC	0.73

🏃 Run Locally

Clone the repository
git clone <your-repo-link>
cd loan-risk-scorer
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run loan_default_app.py

🚀 Future Improvements
Add interactive probability visualization in the UI
Implement SHAP explainability for feature importance
Display model performance dashboard in the app
Add user authentication & logging for production

👩‍💻 Author

Tanisha – Data & ML Enthusiast | end -end ml project 



