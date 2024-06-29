import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Define a function to load data and prepare it
def load_and_prepare_data(train_path, test_path, columns=None):
    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop(columns=['lung'])
    y_train = train_data['lung']
    
    # Load testing data
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=['lung'])
    y_test = test_data['lung']
    
    if columns is not None:
        X_train = X_train[columns]  # Use the same columns used in training
        X_test = X_test[columns]
    
    return X_train, X_test, y_train, y_test

# Define a function to train the model and calculate metrics
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Define and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict probabilities and binary outcomes
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_proba_train > 0.5).astype(int)
    
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_proba_test > 0.5).astype(int)
    
    # Calculate ROC curve and AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    auc_score_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
    auc_score_test = auc(fpr_test, tpr_test)
    
    # Calculate confusion matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    return (fpr_train, tpr_train, auc_score_train, cm_train, y_train, y_pred_train), \
           (fpr_test, tpr_test, auc_score_test, cm_test, y_test, y_pred_test), model, X_train.columns

# Function to calculate additional metrics from confusion matrix
def calculate_metrics(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    informedness = recall + specificity - 1
    dor = (TP / FN) / (FP / TN) if FN > 0 and TN > 0 else 0

    return {
        "Precision": round(precision, ndigits=4),
        "F1-Score": round(f1_score, ndigits=4),
        "Accuracy": round(accuracy, ndigits=4),
        "NPV": round(npv, ndigits=4),
        "MCC": round(mcc, ndigits=4),
        "Informedness": round(informedness, ndigits=4),
        "DOR": round(dor, ndigits=4)
    }

# Load and prepare data
X_train_f, X_test_f, y_train_f, y_test_f = load_and_prepare_data(
    '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv',
    '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv'
)

X_train_m, X_test_m, y_train_m, y_test_m = load_and_prepare_data(
    '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv',
    '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv'
)

# Train and evaluate models
train_results_f, test_results_f, model_f, columns_f = train_and_evaluate(X_train_f, y_train_f, X_test_f, y_test_f)
train_results_m, test_results_m, model_m, columns_m = train_and_evaluate(X_train_m, y_train_m, X_test_m, y_test_m)

# Calculate metrics for combined datasets
metrics_combined_train_f = calculate_metrics(train_results_f[3])
metrics_combined_test_f = calculate_metrics(test_results_f[3])

metrics_combined_train_m = calculate_metrics(train_results_m[3])
metrics_combined_test_m = calculate_metrics(test_results_m[3])

# Print combined metrics
print("Combined Female Training Metrics:", metrics_combined_train_f)
print("Combined Female Testing Metrics:", metrics_combined_test_f)
print("Combined Male Training Metrics:", metrics_combined_train_m)
print("Combined Male Testing Metrics:", metrics_combined_test_m)

# Plot all ROC curves
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(train_results_f[0], train_results_f[1], label=f'Combined Female Training (AUC = {train_results_f[2]:.3f})', color='blue', linestyle='dashed')
plt.plot(test_results_f[0], test_results_f[1], label=f'Combined Female Testing (AUC = {test_results_f[2]:.3f})', color='blue')
plt.plot(train_results_m[0], train_results_m[1], label=f'Combined Male Training (AUC = {train_results_m[2]:.3f})', color='green', linestyle='dashed')
plt.plot(test_results_m[0], test_results_m[1], label=f'Combined Male Testing (AUC = {test_results_m[2]:.3f})', color='green')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Logistic Regression: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/Feature Rich Photos/LogRegTest.png', dpi=300, bbox_inches='tight')
plt.show()




