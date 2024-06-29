import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
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

# Function to train and evaluate model on PLCO and UKB data
def train_evaluate_model(plco_data_path, ukb_data_path):
    # Load and prepare PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco = plco_data.drop(columns=['lung'])
    y_plco = plco_data['lung']
    X_train_plco, X_test_plco, y_train_plco, y_test_plco = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

    # Create and train the GaussianNB model
    model = GaussianNB()
    model.fit(X_train_plco, y_train_plco)

    # Evaluate on PLCO test set
    y_pred_plco_test = model.predict_proba(X_test_plco)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_test_plco, y_pred_plco_test)
    auc_plco = auc(fpr_plco, tpr_plco)
    
    # Calculate metrics for PLCO test data
    plco_test_metrics = calculate_metrics(y_test_plco, y_pred_plco_test)
    
    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_train_plco)[:, 1]
    # Calculate metrics for PLCO training data
    plco_train_metrics = calculate_metrics(y_train_plco, y_pred_plco_train)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_train_plco.columns]
    y_ukb = ukb_data['lung']

    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    
    # Calculate metrics for UKB data
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, plco_test_metrics, ukb_metrics

# Function to combine test metrics
def combine_test_metrics(plco_test_metrics, ukb_metrics):
    combined_metrics = np.mean([plco_test_metrics, ukb_metrics], axis=0)
    return combined_metrics

# Train and evaluate models for male data
male_plco_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_plco_test_metrics, male_ukb_metrics = train_evaluate_model(male_plco_path, male_ukb_path)

# Combine male test metrics
male_combined_test_metrics = combine_test_metrics(male_plco_test_metrics, male_ukb_metrics)

# Train and evaluate models for female data
female_plco_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv'

(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_plco_test_metrics, female_ukb_metrics = train_evaluate_model(female_plco_path, female_ukb_path)

# Combine female test metrics
female_combined_test_metrics = combine_test_metrics(female_plco_test_metrics, female_ukb_metrics)

# Print metrics for male data
print("\nMale Training Metrics:")
print("Precision: ", round(male_plco_train_metrics[0], 4))
print("F1 Score: ", round(male_plco_train_metrics[1], 4))
print("Accuracy: ", round(male_plco_train_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(male_plco_train_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(male_plco_train_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(male_plco_train_metrics[5], 4))
print("Informedness: ", round(male_plco_train_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(male_plco_train_metrics[7], 4))
print("\nMale Combined Testing Metrics:")
print("Precision: ", round(male_combined_test_metrics[0], 4))
print("F1 Score: ", round(male_combined_test_metrics[1], 4))
print("Accuracy: ", round(male_combined_test_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(male_combined_test_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(male_combined_test_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(male_combined_test_metrics[5], 4))
print("Informedness: ", round(male_combined_test_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(male_combined_test_metrics[7], 4))

# Print metrics for female data
print("\nFemale Training Metrics:")
print("Precision: ", round(female_plco_train_metrics[0], 4))
print("F1 Score: ", round(female_plco_train_metrics[1], 4))
print("Accuracy: ", round(female_plco_train_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(female_plco_train_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(female_plco_train_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(female_plco_train_metrics[5], 4))
print("Informedness: ", round(female_plco_train_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(female_plco_train_metrics[7], 4))
print("\nFemale Combined Testing Metrics:")
print("Precision: ", round(female_combined_test_metrics[0], 4))
print("F1 Score: ", round(female_combined_test_metrics[1], 4))
print("Accuracy: ", round(female_combined_test_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(female_combined_test_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(female_combined_test_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(female_combined_test_metrics[5], 4))
print("Informedness: ", round(female_combined_test_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(female_combined_test_metrics[7], 4))

# Plot all ROC curves on the same figure
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/Feature Rich Photos/NaiveBayesFeatureRich.png', dpi=300, bbox_inches='tight')
plt.show()

