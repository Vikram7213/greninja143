import matplotlib
matplotlib.use('Agg')

from pydantic import BaseModel
import joblib
import pandas as pd
import mlflow
import streamlit as st
import os
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

mlflow.set_tracking_uri('http://127.0.0.1:5003')
mlflow.set_experiment('loan_pred')

model_name = 'RF_Loan_model.joblib'
model = joblib.load(model_name)
global_run_id = None

# Define the LoanPred model
class LoanPred(BaseModel):
    Gender: str
    Married: str
    Dependents: float
    Education: str
    Self_Employed: str
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str
    TotalIncome: float

def stream():
    st.title('Loan Prediction')
    st.sidebar.header('4th year project')
    # Collect user input
    gender = st.selectbox('Select Gender', ('Male', 'Female'))
    married = st.selectbox('Marital Status', ('Yes', 'No'))
    dependents = st.select_slider('Enter Number of Dependents', options=[0, 1, 2, 3])
    education = st.radio('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.radio('Self Employed?', ('Yes', 'No'))
    loan_amount = st.number_input('Enter Loan Amount')
    loan_amount_term = st.number_input('Enter Loan Term')
    credit_history = st.radio('Credit History', (1.0, 0.0))
    property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))
    total_income = st.number_input('Enter Total Income')

    if st.button('Predict'):
        loan_pred_data = LoanPred(
            Gender=gender,
            Married=married,
            Dependents=dependents,
            Education=education,
            Self_Employed=self_employed,
            LoanAmount=loan_amount,
            Loan_Amount_Term=loan_amount_term,
            Credit_History=credit_history,
            Property_Area=property_area,
            TotalIncome=total_income
        )

        result = predict_loan_status(loan_pred_data)
        if result == 'Approved':
            st.success(f"your loan is {result}")
        else:
            st.error(f"your loan is {result}")

def metrics(real, probas, got):
    acc = accuracy_score(real, got)
    sc = f1_score(real, got, pos_label='Y')
    fpr, tpr, _ = roc_curve(real, probas[:, 1], pos_label='Y')
    au = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='purple', label='AUC = %0.3f' % au)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/roc.png')
    plt.close()
    
    return acc, sc, au

def mlflow_logging(model, x, y, model_name):
    mlflow.set_tag('mlflow.runName', 'Darkrai!!!!!~')
    probas = model.predict_proba(x)
    pred = model.predict(x)
    accuracy, f1, auc_value = metrics(y, probas, pred)
    
    mlflow.log_params(model.best_params_)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('auc', auc_value)
    mlflow.log_metric('mean_cv_score', model.best_score_)
    
    mlflow.log_artifact('plots/roc.png')
    mlflow.sklearn.log_model(model, model_name)

def predict_loan_status(loan_details: LoanPred):
    global X_test, y_test, model_rf  # Ensure these are in scope
    # Convert the Pydantic model instance to a dictionary and then to a DataFrame
    data = loan_details.dict()
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model_rf.predict(input_data)
    pred_status = 'Approved' if prediction[0] == 'Y' else 'Rejected'
    
    # Log prediction result
    prediction_result = {
        'Timestamp': str(datetime.now()),
        'Prediction': pred_status
    }
    with open('prediction_result.json', 'w') as f:
        json.dump(prediction_result, f)
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow_logging(model_rf, X_test, y_test, 'random_forest')
        mlflow.log_params(data)
        mlflow.log_metric('Prediction', 1 if pred_status == 'Approved' else 0)
        mlflow.log_artifact('prediction_result.json')
        mlflow.end_run()
    
    return pred_status

def train_and_log_models():
    global global_run_id
    data = pd.read_csv("train.csv")

    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

    num = ['TotalIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    cat = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    n_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', FunctionTransformer(func=np.log1p, validate=True))
    ])
    
    c_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', n_transformer, num),
            ('cat', c_transformer, cat)
        ]
    )
    
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=69))
    ])
    
    param_grid_rf = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [6, 10],
    }
    
    cv_model_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='accuracy')
    
    X = data.drop(columns='Loan_Status')
    y = data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    cv_model_rf.fit(X_train, y_train)
    return cv_model_rf, X_test, y_test

if __name__ == '__main__':
    model_rf, X_test, y_test = train_and_log_models() 
    stream()
