'''
This file contains the code to train and evaluate a Random Forest model on the PLCO and UKB data.
The model is trained on the PLCO data, cross-validated on the PLCO data, and tested on the UKB data.
Functions:
    cross_validate_and_train: Performs cross-validation and training on the PLCO data and testing on the UKB data.
        Arguments: plco_data_path, ukb_data_path
    calculate_metrics: Calculates additional metrics for the model.
        Arguments: y_true, y_pred (y_true is the true labels, y_pred is the predicted probabilities of the model)
        Metrics: Precision, F1 Score, Accuracy, Positive Predictive Value (PPV), Negative Predictive Value (NPV), Matthews Correlation Coefficient (MCC), Informedness, Diagnostic Odds Ratio (DOR)
Output:
    - Prints the cross-validation metrics for the model on the PLCO data.
    - Prints the training metrics for the model on the PLCO data.
    - Prints the testing metrics for the model on the UKB data.
    - Saves the ROC curve plot for the model on the PLCO data.
    - Saves the ROC curve plot for the model on the UKB data.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

# Add the top-level directory to the sys.path
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from helper_functions import calculate_metrics, print_metrics, plot_auc

# Function to perform cross-validation and training on PLCO data
def cross_validate_and_train(plco_data_path, ukb_data_path):
    # Load PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']
    
    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train.iloc[train_index], X_plco_train.iloc[val_index]
        y_train, y_val = y_plco_train.iloc[train_index], y_plco_train.iloc[val_index]
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred_val = model.predict_proba(X_val)[:, 1]
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))
    
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)
    
    # Train on the entire PLCO dataset
    model.fit(X_plco_train, y_plco_train)
    
    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_plco_train)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']
    
    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)
    
    return cv_means, cv_stds, (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, ukb_metrics

# Paths to male and female datasets
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

# Perform cross-validation and training for male data
male_cv_means, male_cv_stds, (male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_ukb_metrics = cross_validate_and_train(male_plco_path, male_ukb_path)

# Perform cross-validation and training for female data
female_cv_means, female_cv_stds, (female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_ukb_metrics = cross_validate_and_train(female_plco_path, female_ukb_path)

print_metrics(male_plco_train_metrics, male_ukb_metrics,
              female_plco_train_metrics, female_ukb_metrics,
              male_cv_means, male_cv_stds,
              female_cv_means, female_cv_stds)

plot_auc(male_fpr_plco, male_tpr_plco, male_auc_plco,
             male_fpr_ukb, male_tpr_ukb, male_auc_ukb,
             female_fpr_plco, female_tpr_plco, female_auc_plco,
             female_fpr_ukb, female_tpr_ukb, female_auc_ukb,
             filename='ML Models/Models (Feature Rich)/feature_rich_cross_validation/feature_rich_cv/RandomForestFRCv.png',
             title='Random Forest: ROC Curves for Lung Cancer Prediction',
             linewidth=2)

