
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
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from helper_functions import calculate_metrics, print_metrics, table_of_metrics, plot_roc_curves

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
male_plco_path = 'data_files/imputed/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'data_files/imputed/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'data_files/imputed/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'data_files/imputed/UKB_female_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_ukb_metrics = train_evaluate_model(male_plco_path, male_ukb_path)
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_ukb_metrics = train_evaluate_model(female_plco_path, female_ukb_path)

# print metrics
print_metrics(male_plco_train_metrics, male_ukb_metrics, female_plco_train_metrics, female_ukb_metrics)

# save table of metrics
table_of_metrics('ANN Male Training Metrics', 'ANN', None, None, male_plco_train_metrics)
table_of_metrics('ANN Male Testing Metrics', 'ANN', None, None, male_ukb_metrics)

table_of_metrics('ANN Female Training Metrics', 'ANN', None, None, female_plco_train_metrics)
table_of_metrics('ANN Female Testing Metrics', 'ANN', None, None, female_ukb_metrics)

plot_roc_curves(male_fpr_plco, male_tpr_plco, male_fpr_ukb, male_tpr_ukb, female_fpr_plco, female_tpr_plco, female_fpr_ukb, female_tpr_ukb, male_auc_plco, male_auc_ukb, female_auc_plco, female_auc_ukb, 'ANN Feature Rich: ROC Curves', 'ANN')
