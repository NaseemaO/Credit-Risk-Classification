# Credit-Risk-Classification
Module 20 Challenge. Machine Learning - Supervised - Logistic Regression Model

## Purpose 
To predict whether the loan status of a loan is Healthy or High Risk. 

This is accomplished by building a model that will train and evaluate the dataset of historical lending activity (loans) from a peer-to-peer lending services company to predict loans as either Healthy or High Risk and therby the credit worthiness of borrowers.  

## Input Data
Dataset file: 'lending_data.csv'. 

Data included in the file: 
1. loan_size,
2. interest_rate,
3. borrower_income, 
4. debt_to_income, 
5. num_of_accounts
6. derogatory_marks
7. total_debt
8. loan_status

The input data file contains a total of 77,536 records of loans. 

Value of '0' in the loan_status indicates a Healthy Loan, and '1' a High Risk Loan.

  75036 are Healthy Loans.

  2500 are High Risk Loans.

## Process
#### Import dependencies, libraries, and modules:

    import numpy as np

    import pandas as pd

    from pathlib import Path

    from sklearn.metrics import confusion_matrix, classification_report

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score  

#### Loading
Load input .csv file into a Pandas Dataframe

#### Separate data
X for features, and 
y for the target label. 

The first 7 data 'columns' in input file are features, and the loan_status is the variable to make predictions. 

#### Split data
Data further separated into Training Data and Test Data using the train_test_split. 

  Training Data: 58152 rows, 7 columns
  Test Data: 19384 rows, 7 columns

#### Prediction Model
Logistic Regression model (Supervised Machine Learning) used to make predictions and get the Data Scores.

Training Data Score: 0.9914

Test Data Score: 0.9924 

#### Evaluation
Evaluated the Logistic Regression model's performance on the Test Data by 
  Generating a confusion matrix, and
  Classification Report

## Results
Machine Learning Model Logistic Regression performed on the Test Data

Loans with '0' value are Healthy Loans. This is the Negative '0' value.
Loans with '1' value are High Risk Loans. This is the Positive '1' value. 

* Logistic Regression Prediction Accuracy Score: 0.9924 

* Confusion Matrix Results: 
  array ([[18679,  80],
          [67,   558]])

  ** True Negative: 118679 
    False Positive: 80
    These are the Healthy Loans. Total: 18759

  ** False Negative: 67
    True Positive: 558 
    These are the High Risk Loans. Total: 625

* Classification Report Results:

  ** The prediction on the Healthy Loans is a 100%.  
  Precision is 1.00 that is 100% (of all the '0' how many are '0'), and 
  Recall is 1.00 that is 100%  (of all that are '0' how many did we predict were '0') 

  ** The prediction High Risk Loans '1':
  Precision: 87%, and 
  Recall: 89% Recall. 
  These are pretty close. 

  ** Total Healthy Loans: 18759.  

  ** Total High Risk Loans: 625.  

  ** Total Loans in Test Data: 19384

  ** The Overal accuracy is 99%, and 

  ** Weighted average is 99%

  ** Fl-score, is a balanced mean between precision and recall. 

## Summary
* The Logistic Regression Model predicted Test Data Accuracy Score: 0.9924 

* The Confusion Matrix and Classification Report Predicted:

  100% Precision and Recall on Healthy Loans,

  87% Precision and 89% Recall on High Risk Loans. 

In comparison, the Precision and Recall on the High Risk Loans is about 10% less than the Logistic Regression data accuracy score. However the overall score on both the Healthy and the High Risk Loans is good at 99%


