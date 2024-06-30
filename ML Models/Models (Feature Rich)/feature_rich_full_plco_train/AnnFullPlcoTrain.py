'''
This file contains the code to train and evaluate an Artificial Neural Network (ANN) model on the PLCO and UKB data.
The model is trained on all the PLCO data, validated on the PLCO data, and tested on the UKB data.
Functions: 
    train_evaluate_model: Trains and evaluates the ANN model on the PLCO and UKB data.
        Arguments: plco_data_path, ukb_data_path
    calculate_metrics: Calculates additional metrics for the model.
        Arguments: y_true, y_pred (y_true is the true labels, y_pred is the predicted probabilities of the model)
        Metrics: Precision, F1 Score, Accuracy, Positive Predictive Value (PPV), Negative Predictive Value (NPV), Matthews Correlation Coefficient (MCC), Informedness, Diagnostic Odds Ratio (DOR)
Output: 
    - Prints the metrics for the model on the PLCO training data and the UKB data.
    - Saves the ROC curve plot for the model on the PLCO training data.
    - Saves the ROC curve plot for the model on the UKB data.
    - Saves the metrics for the model on the PLCO training data and the UKB data.
    - Saves the model summary.
    - Saves the model.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
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

# Function to train and evaluate model on PLCO and UKB data
def train_evaluate_model(plco_data_path, ukb_data_path):
    # Load and prepare PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']

    # Define the model
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_plco_train.shape[1]],
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(80, activation='relu',
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(1)
        ])

    # Print the model summary
    print(model.summary())

    # Compile the model
    model.compile(
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),]
    )

    # Train the model on PLCO training data
    model.fit(X_plco_train, y_plco_train, epochs=10, batch_size=1024, verbose=2)
    
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
    
    # Calculate metrics for UKB data
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, ukb_metrics

# Paths to male and female datasets
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_ukb_metrics = train_evaluate_model(male_plco_path, male_ukb_path)
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_ukb_metrics = train_evaluate_model(female_plco_path, female_ukb_path)

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

# Combined Data
print("\nMale Testing Metrics:")
print("Precision: ", round(male_ukb_metrics[0], 4))
print("F1 Score: ", round(male_ukb_metrics[1], 4))
print("Accuracy: ", round(male_ukb_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(male_ukb_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(male_ukb_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(male_ukb_metrics[5], 4))
print("Informedness: ", round(male_ukb_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(male_ukb_metrics[7], 4))

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

# Combined data
print("\nFemale Testing Metrics:")
print("Precision: ", round(female_ukb_metrics[0], 4))
print("F1 Score: ", round(female_ukb_metrics[1], 4))
print("Accuracy: ", round(female_ukb_metrics[2], 4))
print("Positive Predictive Value (PPV): ", round(female_ukb_metrics[3], 4))
print("Negative Predictive Value (NPV): ", round(female_ukb_metrics[4], 4))
print("Matthews Correlation Coefficient (MCC): ", round(female_ukb_metrics[5], 4))
print("Informedness: ", round(female_ukb_metrics[6], 4))
print("Diagnostic Odds Ratio (DOR): ", round(female_ukb_metrics[7], 4))

# Plot all ROC curves on the same figure
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network: ROC Curves for Lung Cancer Prediction (FT)')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/feature_rich_full_plco_train/feature_rich_full_train_photos/ANNFeatureRichFullTrain.png', dpi=300, bbox_inches='tight')
plt.show()
