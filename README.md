# Student Performance Predictor (ML Project)

## Description
This is a Flask web application that uses Machine Learning to predict a student's performance index.
Users input data such as study hours, sleep hours, previous scores, participation in activities and number of practiced questions, and the trained model outputs a predicted performance index.

---

## Technologies
- Python 3.9+
- Flask (web application framework)
- pandas, numpy (data handling)
- scikit-learn, scipy (machine learning)
- matplotlib, seaborn (for data analysis)
- dill (for saving)
- kaggle (for dataset download)

---

## Features
- Data Ingestion - load and split dataset
- Data Transformation - preprocessiong and feature enggineering
- Model Training - train and evaluate ML models
- Prediction Pipeline - reusable pipeline for inference
- Flask Web App - interactive UI for predictions

---

## Installation and Running

1. Clone the repository:

git clone https://github.com/Hanna-hk/Student_Performance_ML_project.git
cd Student_Performance_ML_project

2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

3. Install dependencies:

pip install -r requirements.txt

4. Run the Flask server:

python app.py

5. Open in your browser:

http://127.0.0.1:5000/
