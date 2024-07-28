'''
This file contains the code to train and evaluate a Logistic Regression model on the PLCO and UKB data.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Function to perform cross-validation and training on PLCO data
def cross_validate_and_train(plco_data_path, ukb_data_path, solver='lbfgs', max_iter=50000):
    # Load PLCO data
    plco_data = pd.read_csv(plco_data_path)
    ukb_data = pd.read_csv(ukb_data_path)
    
    # Ensure both datasets have the same features
    common_features = list(set(plco_data.columns) & set(ukb_data.columns))
    common_features.remove('lung')
    
    X_plco_train = plco_data[common_features].values
    y_plco_train = plco_data['lung'].values
    
    # Scale the data
    scaler = StandardScaler()
    X_plco_train = scaler.fit_transform(X_plco_train)
    
    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []
    loss_curves = []

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train[train_index], X_plco_train[val_index]
        y_train, y_val = y_plco_train[train_index], y_plco_train[val_index]
        
        model = LogisticRegression(max_iter=max_iter, solver=solver, verbose=1)
        model.fit(X_train, y_train)
        y_pred_val = model.predict_proba(X_val)[:, 1]
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))

        # Calculate log loss for each iteration
        losses = []
        for i in range(1, model.n_iter_[0] + 1):
            model_iter = LogisticRegression(max_iter=i, solver=solver, warm_start=True)
            model_iter.fit(X_train, y_train)
            y_pred_iter = model_iter.predict_proba(X_val)[:, 1]
            losses.append(log_loss(y_val, y_pred_iter))
        loss_curves.append(losses)
    
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

    # Prepare UKB data
    X_ukb = ukb_data[common_features].values
    X_ukb = scaler.transform(X_ukb)  # Apply the same scaling
    y_ukb = ukb_data['lung'].values
    
    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)
    
    return cv_means, cv_stds, (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, ukb_metrics, loss_curves

# Paths to male and female datasets
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

# Perform cross-validation and training for male data
male_cv_means, male_cv_stds, (male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_ukb_metrics, male_loss_curves = cross_validate_and_train(male_plco_path, male_ukb_path, solver='lbfgs', max_iter=50000)

# Perform cross-validation and training for female data
female_cv_means, female_cv_stds, (female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_ukb_metrics, female_loss_curves = cross_validate_and_train(female_plco_path, female_ukb_path, solver='lbfgs', max_iter=50000)

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

# Plot loss curves for each fold in cross-validation
plt.figure(figsize=(10, 5))
for i, losses in enumerate(male_loss_curves):
    plt.plot(losses, label=f'Male Fold {i+1}', alpha=0.3)
for i, losses in enumerate(female_loss_curves):
    plt.plot(losses, label=f'Female Fold {i+1}', alpha=0.3)

plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Logistic Regression: Loss Curves for Cross-Validation Folds')
plt.legend(loc='upper right', fontsize='small', framealpha=0.5)
plt.savefig('ML Models/Models (Feature Rich)/feature_rich_cross_validation/feature_rich_cv/LogRegFeatureRichLossCurves.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot ROC curves for UKB data
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})', linewidth=2)
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})', linewidth=2)
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})', linewidth=2)
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/feature_rich_cross_validation/feature_rich_cv/LogRegFeatureRichROCCurves.png', dpi=300, bbox_inches='tight')
plt.show()

