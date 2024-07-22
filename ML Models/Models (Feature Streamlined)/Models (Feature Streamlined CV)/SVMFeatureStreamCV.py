import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix

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

# Function to perform k-fold cross validation, evaluation on PLCO data, and test on UKB and NHIS
def cross_validate_and_train(plco_data_path, ukb_data_path, nhis_data_path):
    # Load all the data (PLCO, UKB, NHIS)
    plco_data = pd.read_csv(plco_data_path)
    ukb_data = pd.read_csv(ukb_data_path)
    nhis_data = pd.read_csv(nhis_data_path)

    # Create variables for features
    X_plco = plco_data.drop(columns=['lung'])
    X_ukb = ukb_data.drop(columns=['lung'])
    X_nhis = nhis_data.drop(columns=['lung'])

    # Determine common features between the datasets
    common_features = X_nhis.columns.intersection(X_plco.columns).intersection(X_ukb.columns)
    X_plco = X_plco[common_features]
    X_ukb = X_ukb[common_features]
    X_nhis = X_nhis[common_features]

    # Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []

    for train_index, val_index in kf.split(X_plco):
        X_train, X_val = X_plco.iloc[train_index], X_plco.iloc[val_index]
        y_train, y_val = plco_data['lung'].iloc[train_index], plco_data['lung'].iloc[val_index]

        model = SVC(probability=True, random_state=42)
        model.fit(X_train, y_train)
        y_pred_val = model.predict_proba(X_val)[:, 1]
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))
    
    # Cross validation metrics
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)

    # Train on the entire PLCO dataset
    model = SVC(probability=True, random_state=42)
    model.fit(X_plco, plco_data['lung'])

    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_plco)[:, 1]

    # Calculate metrics for PLCO training data
    plco_train_metrics = calculate_metrics(plco_data['lung'], y_pred_plco_train)

    # Plot the ROC curve for the training set
    fpr_plco, tpr_plco, _ = roc_curve(plco_data['lung'], y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)

    # Check how many people have lung cancer in each data set
    print(f"PLCO: {plco_data['lung'].value_counts().get(1, 0)}")
    print(f"UKB: {ukb_data['lung'].value_counts().get(1, 0)}")
    print(f"NHIS: {nhis_data['lung'].value_counts().get(1, 0)}")

    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(ukb_data['lung'], y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(ukb_data['lung'], y_pred_ukb)

    # Predict on NHIS data
    y_pred_nhis = model.predict_proba(X_nhis)[:, 1]
    fpr_nhis, tpr_nhis, _ = roc_curve(nhis_data['lung'], y_pred_nhis)
    auc_nhis = auc(fpr_nhis, tpr_nhis)
    nhis_metrics = calculate_metrics(nhis_data['lung'], y_pred_nhis)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), (fpr_nhis, tpr_nhis, auc_nhis), cv_means, cv_stds, plco_train_metrics, ukb_metrics, nhis_metrics

# Paths to male and female datasets
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_nhis_path = 'Input/male_filtered_chosen_NHIS_imputed.csv'

female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_nhis_path = 'Input/female_filtered_chosen_NHIS_imputed.csv'

# Perform cross-validation and training for male data
(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), (male_fpr_nhis, male_tpr_nhis, male_auc_nhis), male_cv_means, male_cv_stds, male_plco_train_metrics, male_ukb_metrics, male_nhis_metrics = cross_validate_and_train(male_plco_path, male_ukb_path, male_nhis_path)

# Perform cross-validation and training for female data
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), (female_fpr_nhis, female_tpr_nhis, female_auc_nhis), female_cv_means, female_cv_stds, female_plco_train_metrics, female_ukb_metrics, female_nhis_metrics = cross_validate_and_train(female_plco_path, female_ukb_path, female_nhis_path)

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

# Print validation metrics for male data
print("\nMale Training Metrics:")
print("Precision: ", round(male_plco_train_metrics[0], 4))
print("F1 Score: ", round(male_plco_train_metrics[1], 4))
print("Accuracy: ", round(male_plco_train_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(male_plco_train_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(male_plco_train_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(male_plco_train_metrics[5], 4))
print("Informedness: ", round(male_plco_train_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(male_plco_train_metrics[7], 4))

# Print testing results for male data (UKB)
print("\nMale Testing Metrics on UKB Data:")
print(f"Precision: {male_ukb_metrics[0]:.4f}")
print(f"F1 Score: {male_ukb_metrics[1]:.4f}")
print(f"Accuracy: {male_ukb_metrics[2]:.4f}")
print(f"PPV: {male_ukb_metrics[3]:.4f}")
print(f"NPV: {male_ukb_metrics[4]:.4f}")
print(f"MCC: {male_ukb_metrics[5]:.4f}")
print(f"Informedness: {male_ukb_metrics[6]:.4f}")
print(f"DOR: {male_ukb_metrics[7]:.4f}")

# Print testing results for male data (NHIS)
print("\nMale Testing Metrics on NHIS Data:")
print(f"Precision: {male_nhis_metrics[0]:.4f}")
print(f"F1 Score: {male_nhis_metrics[1]:.4f}")
print(f"Accuracy: {male_nhis_metrics[2]:.4f}")
print(f"PPV: {male_nhis_metrics[3]:.4f}")
print(f"NPV: {male_nhis_metrics[4]:.4f}")
print(f"MCC: {male_nhis_metrics[5]:.4f}")
print(f"Informedness: {male_nhis_metrics[6]:.4f}")
print(f"DOR: {male_nhis_metrics[7]:.4f}")

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

# Print validation for female data
print("\nFemale Training Metrics:")
print("Precision: ", round(female_plco_train_metrics[0], 4))
print("F1 Score: ", round(female_plco_train_metrics[1], 4))
print("Accuracy: ", round(female_plco_train_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(female_plco_train_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(female_plco_train_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(female_plco_train_metrics[5], 4))
print("Informedness: ", round(female_plco_train_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(female_plco_train_metrics[7], 4))

# Print testing results for female data (UKB)
print("\nFemale Testing Metrics on UKB Data:")
print(f"Precision: {female_ukb_metrics[0]:.4f}")
print(f"F1 Score: {female_ukb_metrics[1]:.4f}")
print(f"Accuracy: {female_ukb_metrics[2]:.4f}")
print(f"PPV: {female_ukb_metrics[3]:.4f}")
print(f"NPV: {female_ukb_metrics[4]:.4f}")
print(f"MCC: {female_ukb_metrics[5]:.4f}")
print(f"Informedness: {female_ukb_metrics[6]:.4f}")
print(f"DOR: {female_ukb_metrics[7]:.4f}")

# Print testing results for female data (NHIS)
print("\nFemale Testing Metrics on NHIS Data:")
print(f"Precision: {female_nhis_metrics[0]:.4f}")
print(f"F1 Score: {female_nhis_metrics[1]:.4f}")
print(f"Accuracy: {female_nhis_metrics[2]:.4f}")
print(f"PPV: {female_nhis_metrics[3]:.4f}")
print(f"NPV: {female_nhis_metrics[4]:.4f}")
print(f"MCC: {female_nhis_metrics[5]:.4f}")
print(f"Informedness: {female_nhis_metrics[6]:.4f}")
print(f"DOR: {female_nhis_metrics[7]:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
plt.plot(male_fpr_nhis, male_tpr_nhis, label=f'NHIS Male (AUC = {male_auc_nhis:.3f})')
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
plt.plot(female_fpr_nhis, female_tpr_nhis, label=f'NHIS Female (AUC = {female_auc_nhis:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM: ROC Curves for Lung Cancer Prediction (chosen) CV')
plt.legend(loc='lower right')
plt.savefig('/Users/teresanguyen/Lung-Statistical-Biopsy/ML Models/Models (Feature Streamlined)/Models (Feature Streamlined CV)/feature streamlined cv photos/SVMFStreamlinedCvChosen.png', dpi=300, bbox_inches='tight')
plt.show()
