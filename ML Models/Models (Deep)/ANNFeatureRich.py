import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

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
    y_pred_test = model.predict(X_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    auc_test = auc(fpr_test, tpr_test)

    # Return results and model
    return fpr_test, tpr_test, auc_test, model, X_train.columns

# Process each dataset
fpr_plco_f, tpr_plco_f, auc_plco_f, model_f, columns_f = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv')
fpr_plco_m, tpr_plco_m, auc_plco_m, model_m, columns_m = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv')

# Process UKB using trained models and same features used in PLCO training
fpr_ukb_f, tpr_ukb_f, auc_ukb_f, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv', columns_f)
fpr_ukb_m, tpr_ukb_m, auc_ukb_m, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv', columns_m)

# Plot all ROC curves
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(fpr_plco_f, tpr_plco_f, label=f'PLCO Female (AUC = {auc_plco_f:.3f})')
plt.plot(fpr_plco_m, tpr_plco_m, label=f'PLCO Male (AUC = {auc_plco_m:.3f})')
plt.plot(fpr_ukb_f, tpr_ukb_f, label=f'UKB Female (AUC = {auc_ukb_f:.3f})')
plt.plot(fpr_ukb_m, tpr_ukb_m, label=f'UKB Male (AUC = {auc_ukb_m:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Deep)/Feature Rich Photos/ANNFeatureRich.png', dpi=300, bbox_inches='tight')
plt.show()
