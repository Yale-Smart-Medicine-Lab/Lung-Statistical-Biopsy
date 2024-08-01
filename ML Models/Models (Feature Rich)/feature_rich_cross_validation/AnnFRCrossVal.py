'''
This file contains the code to train and evaluate an Artificial Neural Network (ANN) model on the PLCO and UKB data.
Functions:
    calculate_metrics: Calculates additional metrics for the model.
        Arguments: y_true, y_pred (y_true is the true labels, y_pred is the predicted probabilities of the model)
        Metrics: Precision, F1 Score, Accuracy, Positive Predictive Value (PPV), Negative Predictive Value (NPV), Matthews Correlation Coefficient (MCC), Informedness, Diagnostic Odds Ratio (DOR)
    create_model: Defines the ANN model architecture.
        Arguments: input_shape
    Cross_validate_and_train: Performs cross-validation and training on the PLCO data.
        Arguments: plco_data_path, ukb_data_path
Output:
    - Prints the metrics for the model on the PLCO training data and the UKB data.
    - Saves the ROC curve plot for the model on the PLCO training data.
    - Saves the ROC curve plot for the model on the UKB data.
    - Saves the metrics for the model on the PLCO training data and the UKB data.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
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

# Function to define the model
def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[input_shape],
                           kernel_initializer=keras.initializers.glorot_normal(),
                           bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(80, activation='relu',
                           kernel_initializer=keras.initializers.glorot_normal(),
                           bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

# Function to perform cross-validation and training on PLCO data
def cross_validate_and_train(plco_data_path, ukb_data_path):
    # Load PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']
    
    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []
    history_list = []  # New list to store the history of each fold

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train.iloc[train_index], X_plco_train.iloc[val_index]
        y_train, y_val = y_plco_train.iloc[train_index], y_plco_train.iloc[val_index]
        
        model = create_model(X_plco_train.shape[1])
        history = model.fit(X_train, y_train, epochs=10, batch_size=1024, verbose=0, validation_data=(X_val, y_val))
        y_pred_val = model.predict(X_val)
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))
        history_list.append(history)
    
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)
    
    # Train on the entire PLCO dataset
    model = create_model(X_plco_train.shape[1])
    history = model.fit(X_plco_train, y_plco_train, epochs=10, batch_size=1024, verbose=2, validation_split=0.1)

    # Evaluate on PLCO training set
    model.evaluate(X_plco_train, y_plco_train, verbose=2, batch_size=1024)
    y_pred_plco_train = model.predict(X_plco_train)
    
    # Calculate metrics for PLCO training data
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    # Plot the ROC curve for the validation set
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']
    
    # Predict on UKB data
    y_pred_ukb = model.predict(X_ukb)
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)
    
    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), cv_means, cv_stds, plco_train_metrics, ukb_metrics, history_list, history  # Return history

# Paths to male and female datasets
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

# Perform cross-validation and training for male data
(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_cv_means, male_cv_stds, male_plco_train_metrics, male_ukb_metrics, male_history_list, male_history = cross_validate_and_train(male_plco_path, male_ukb_path)

# Perform cross-validation and training for female data
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_cv_means, female_cv_stds, female_plco_train_metrics, female_ukb_metrics, female_history_list, female_history = cross_validate_and_train(female_plco_path, female_ukb_path)

# Plot the loss curves for male data
plt.figure(figsize=(10, 5))
for history in male_history_list:  # Plot loss for each fold
    plt.plot(history.history['loss'], 'b-', alpha=0.3)
    plt.plot(history.history['val_loss'], 'r-', alpha=0.3)
plt.title('ANN Feature Rich: Male Data Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# Plot the loss curves for female data
plt.figure(figsize=(10, 5))
for history in female_history_list:  # Plot loss for each fold
    plt.plot(history.history['loss'], 'b-', alpha=0.3)
    plt.plot(history.history['val_loss'], 'r-', alpha=0.3)
plt.title('ANN Feature Rich: Female Data Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

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

# Print testing results for male data
print("\nMale Testing Metrics on UKB Data:")
print(f"Precision: {male_ukb_metrics[0]:.4f}")
print(f"F1 Score: {male_ukb_metrics[1]:.4f}")
print(f"Accuracy: {male_ukb_metrics[2]:.4f}")
print(f"PPV: {male_ukb_metrics[3]:.4f}")
print(f"NPV: {male_ukb_metrics[4]:.4f}")
print(f"MCC: {male_ukb_metrics[5]:.4f}")
print(f"Informedness: {male_ukb_metrics[6]:.4f}")
print(f"DOR: {male_ukb_metrics[7]:.4f}")

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

# Print testing results for female data
print("\nFemale Testing Metrics on UKB Data:")
print(f"Precision: {female_ukb_metrics[0]:.4f}")
print(f"F1 Score: {female_ukb_metrics[1]:.4f}")
print(f"Accuracy: {female_ukb_metrics[2]:.4f}")
print(f"PPV: {female_ukb_metrics[3]:.4f}")
print(f"NPV: {female_ukb_metrics[4]:.4f}")
print(f"MCC: {female_ukb_metrics[5]:.4f}")
print(f"Informedness: {female_ukb_metrics[6]:.4f}")
print(f"DOR: {female_ukb_metrics[7]:.4f}")

# Plot all ROC curves on the same figure
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network: ROC Curves for Lung Cancer Prediction CV')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/feature_rich_cross_validation/feature_rich_cv/AnnFrCv.png', dpi=300, bbox_inches='tight')
plt.show()
