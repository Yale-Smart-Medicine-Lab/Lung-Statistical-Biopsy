#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur 13 July 2024

@author: Gregory Hart
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix

# Function to calculate additional metrics
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    
    precision = precision_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    accuracy = accuracy_score(y_true, y_pred_labels)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred_labels)
    informedness = (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0
    dor = (tp / fn) / (fp / tn) if fn > 0 and fp > 0 and tn > 0 else 0

    return precision, f1, accuracy, ppv, npv, mcc, informedness, dor

# Function to train and evaluate model on PLCO and UKB data
def train_evaluate_model(plco_data_path, ukb_data_path, model_type, title, filename):
    # Load and prepare PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']

    # Create and train the model
    if model_type == SVC:
        model = model_type(probability=True)
    else:
        model = model_type()
    model.fit(X_plco_train, y_plco_train)
    
    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_plco_train)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)

    # Calculate metrics for PLCO training data
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']

    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    
    # Calculate metrics for UKB data
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    # I would like to plot the loss curve, but I don't know if every model type has a loss_curve_ attribute
    # plt.figure(figsize=(10, 10))
    # plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    # plt.plot(model.loss_curve_)
    # plt.xlabel('Epochs')
    # # plt.ylabel('Loss/Classification Score')
    # plt.title(title)
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, ukb_metrics

# Plot all ROC curves on the same figure
def plot_auc(male_fpr_plco, male_tpr_plco, male_auc_plco,
             male_fpr_ukb, male_tpr_ukb, male_auc_ukb,
             female_fpr_plco, female_tpr_plco, female_auc_plco,
             female_fpr_ukb, female_tpr_ukb, female_auc_ukb,
             filename, title, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})', **kwargs)
    plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})', **kwargs)
    plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})', **kwargs)
    plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})', **kwargs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Print metrics
def print_metrics(male_plco_train_metrics, male_ukb_metrics,
                  female_plco_train_metrics, female_ukb_metrics,
                  male_cv_means=None, male_cv_stds=None,
                  female_cv_means=None, female_cv_stds=None):
    if male_cv_means is not None:
        # Print cross-validation results for male data
        print("\nMale Cross-Validation Metrics (Mean ± Std):")
        print(f"Precision: {male_cv_means[0]:.4f} ± {male_cv_stds[0]:.4f}")
        print(f"F1 Score: {male_cv_means[1]:.4f} ± {male_cv_stds[1]:.4f}")
        print(f"Accuracy: {male_cv_means[2]:.4f} ± {male_cv_stds[2]:.4f}")
        print(f"PPV: {male_cv_means[3]:.4f} ± {male_cv_stds[3]:.4f}")
        print(f"NPV: {male_cv_means[4]:.4f} ± {male_cv_stds[4]:.4f}")
        print(f"MCC: {male_cv_means[5]:.4f} ± {male_cv_stds[5]:.4f}")
        print(f"Informedness: {male_cv_means[6]:.4f} ± {male_cv_stds[6]:.4f}")
        print(f"DOR: {male_cv_means[7]:.4f} ± {male_cv_stds[7]:.4f}")

    # Print training results for male data
    print("\nMale Training Metrics:")
    print("Precision: ", round(male_plco_train_metrics[0], 4))
    print("F1 Score: ", round(male_plco_train_metrics[1], 4))
    print("Accuracy: ", round(male_plco_train_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(male_plco_train_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(male_plco_train_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(male_plco_train_metrics[5], 4))
    print("Informedness: ", round(male_plco_train_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(male_plco_train_metrics[7], 4))

    # Print testing results for male data
    print("\nMale Testing Metrics on UKB Data:")
    print("Precision: ", round(male_ukb_metrics[0], 4))
    print("F1 Score: ", round(male_ukb_metrics[1], 4))
    print("Accuracy: ", round(male_ukb_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(male_ukb_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(male_ukb_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(male_ukb_metrics[5], 4))
    print("Informedness: ", round(male_ukb_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(male_ukb_metrics[7], 4))

    if female_cv_means is not None:
        # Print cross-validation results for female data
        print("\nFemale Cross-Validation Metrics (Mean ± Std):")
        print(f"Precision: {female_cv_means[0]:.4f} ± {female_cv_stds[0]:.4f}")
        print(f"F1 Score: {female_cv_means[1]:.4f} ± {female_cv_stds[1]:.4f}")
        print(f"Accuracy: {female_cv_means[2]:.4f} ± {female_cv_stds[2]:.4f}")
        print(f"PPV: {female_cv_means[3]:.4f} ± {female_cv_stds[3]:.4f}")
        print(f"NPV: {female_cv_means[4]:.4f} ± {female_cv_stds[4]:.4f}")
        print(f"MCC: {female_cv_means[5]:.4f} ± {female_cv_stds[5]:.4f}")
        print(f"Informedness: {female_cv_means[6]:.4f} ± {female_cv_stds[6]:.4f}")
        print(f"DOR: {female_cv_means[7]:.4f} ± {female_cv_stds[7]:.4f}")

    # Print training results for female data
    print("\nFemale Training Metrics:")
    print("Precision: ", round(female_plco_train_metrics[0], 4))
    print("F1 Score: ", round(female_plco_train_metrics[1], 4))
    print("Accuracy: ", round(female_plco_train_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(female_plco_train_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(female_plco_train_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(female_plco_train_metrics[5], 4))
    print("Informedness: ", round(female_plco_train_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(female_plco_train_metrics[7], 4))

    # Print testing results for female data
    print("\nFemale Testing Metrics on UKB Data:")
    print("Precision: ", round(female_ukb_metrics[0], 4))
    print("F1 Score: ", round(female_ukb_metrics[1], 4))
    print("Accuracy: ", round(female_ukb_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(female_ukb_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(female_ukb_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(female_ukb_metrics[5], 4))
    print("Informedness: ", round(female_ukb_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(female_ukb_metrics[7], 4))