import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, confusion_matrix

# Define a function to prepare data, train a model, and return predictions along with true values
def process_and_predict(data_path, columns=None):
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['lung'])
    y = data['lung']

    if columns is not None:
        X = X[columns]  # Ensure using only the same columns used in training
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Define the model
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=1024, verbose=0)

    # Predict and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_train_binary = (y_pred_train > 0.5).astype(int)
    y_pred_test_binary = (y_pred_test > 0.5).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)
    
    cm_train = confusion_matrix(y_train, y_pred_train_binary)
    cm_test = confusion_matrix(y_test, y_pred_test_binary)

    return (fpr_train, tpr_train, auc_train, cm_train, y_train, y_pred_train_binary), \
           (fpr_test, tpr_test, auc_test, cm_test, y_test, y_pred_test_binary), model, X_train.columns

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

# Function to combine predictions and true values
def combine_predictions(results1, results2):
    y_true_combined = np.concatenate([results1[4], results2[4]])
    y_pred_combined = np.concatenate([results1[5], results2[5]])
    cm_combined = confusion_matrix(y_true_combined, y_pred_combined)
    return cm_combined, y_true_combined, y_pred_combined

# Process each dataset
train_results_f, test_results_f, model_f, columns_f = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv')
train_results_m, test_results_m, model_m, columns_m = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv')

# Process UKB using trained models and same features used in PLCO training
train_results_ukb_f, test_results_ukb_f, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv', columns_f)
train_results_ukb_m, test_results_ukb_m, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv', columns_m)

# Combine PLCO and UKB results for training and testing separately
cm_combined_train_f, y_true_combined_train_f, y_pred_combined_train_f = combine_predictions(train_results_f, train_results_ukb_f)
cm_combined_test_f, y_true_combined_test_f, y_pred_combined_test_f = combine_predictions(test_results_f, test_results_ukb_f)

cm_combined_train_m, y_true_combined_train_m, y_pred_combined_train_m = combine_predictions(train_results_m, train_results_ukb_m)
cm_combined_test_m, y_true_combined_test_m, y_pred_combined_test_m = combine_predictions(test_results_m, test_results_ukb_m)

# Calculate metrics for combined training and testing datasets
metrics_combined_train_f = calculate_metrics(cm_combined_train_f)
metrics_combined_test_f = calculate_metrics(cm_combined_test_f)

metrics_combined_train_m = calculate_metrics(cm_combined_train_m)
metrics_combined_test_m = calculate_metrics(cm_combined_test_m)

# Print combined metrics
print("Combined Female Training Metrics:", metrics_combined_train_f)
print("Combined Female Testing Metrics:", metrics_combined_test_f)
print("Combined Male Training Metrics:", metrics_combined_train_m)
print("Combined Male Testing Metrics:", metrics_combined_test_m)


'''Plotting
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(train_results_f[0], train_results_f[1], label=f'PLCO Female Train (AUC = {train_results_f[2]:.3f})')
plt.plot(test_results_f[0], test_results_f[1], label=f'PLCO Female Test (AUC = {test_results_f[2]:.3f})')
plt.plot(train_results_m[0], train_results_m[1], label=f'PLCO Male Train (AUC = {train_results_m[2]:.3f})')
plt.plot(test_results_m[0], test_results_m[1], label=f'PLCO Male Test (AUC = {test_results_m[2]:.3f})')
plt.plot(train_results_ukb_f[0], train_results_ukb_f[1], label=f'UKB Female Train (AUC = {train_results_ukb_f[2]:.3f})')
plt.plot(test_results_ukb_f[0], test_results_ukb_f[1], label=f'UKB Female Test (AUC = {test_results_ukb_f[2]:.3f})')
plt.plot(train_results_ukb_m[0], train_results_ukb_m[1], label=f'UKB Male Train (AUC = {train_results_ukb_m[2]:.3f})')
plt.plot(test_results_ukb_m[0], test_results_ukb_m[1], label=f'UKB Male Test (AUC = {test_results_ukb_m[2]:.3f})')
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Neural Network: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/Feature Rich Photos/ANNMetrics.png', dpi=300, bbox_inches='tight')
plt.show()
'''


