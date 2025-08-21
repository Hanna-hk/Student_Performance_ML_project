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

## Project Structure
FIRST_ML_PROJECT/
│── notebook/                           # Jupyter notebooks for EDA & model training
│   ├── data/
│   │   └── Student_Performance.csv     # Dataset
│   ├── EDA_STUDENT_PERFORMANCE.ipynb   # Exploratory Data Analysis
│   └── MODEL_TRAINING.ipynb            # Model training & experiments
│
│── src/first_ml_project/               # Main ML project source code
│   ├── components/                     # Core ML pipeline components
│   │   ├── data_ingestion.py           # Load & split dataset
│   │   ├── data_transformation.py      # Preprocess & feature engineering
│   │   └── model_trainer.py            # Train & evaluate models
│   │
│   ├── pipeline/                       # ML pipeline scripts
│   │   └── predict_pipeline.py         # Handle predictions
│   │
│   ├── exception.py                    # Custom exception handling
│   ├── logger.py                       # Logging utility
│   └── utils.py                        # Helper functions
│
│── templates/                          # Flask HTML templates
│   ├── home.html
│   └── index.html
│
│── app.py                              # Flask app for deployment
│── README.md                           # Project documentation
│── requirements.txt                    # dependencies
│── .gitignore
│── setup.py
│── venv/                               # Virtual environment

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