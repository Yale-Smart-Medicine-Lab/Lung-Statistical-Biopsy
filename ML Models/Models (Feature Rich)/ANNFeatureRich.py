import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Function to train and evaluate model on PLCO and UKB data
def train_evaluate_model(plco_data_path, ukb_data_path, model_name):
    # Load and prepare PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco = plco_data.drop(columns=['lung'])
    y_plco = plco_data['lung']
    X_train_plco, X_test_plco, y_train_plco, y_test_plco = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

    # Define the model
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_train_plco.shape[1]]),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # Train the model on PLCO training data
    model.fit(X_train_plco, y_train_plco, epochs=10, batch_size=1024, verbose=2)

    # Evaluate on PLCO test set
    y_pred_plco = model.predict(X_test_plco)
    fpr_plco, tpr_plco, _ = roc_curve(y_test_plco, y_pred_plco)
    auc_plco = auc(fpr_plco, tpr_plco)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_train_plco.columns]
    y_ukb = ukb_data['lung']

    # Predict on UKB data
    y_pred_ukb = model.predict(X_ukb)
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb)

# Train and evaluate models for male data
male_plco_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb) = train_evaluate_model(male_plco_path, male_ukb_path, 'Male')

# Train and evaluate models for female data
female_plco_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = '/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv'

(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb) = train_evaluate_model(female_plco_path, female_ukb_path, 'Female')

# Plot all ROC curves on the same figure
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Feature Rich)/Feature Rich Photos/ANNFeatureRich.png', dpi=300, bbox_inches='tight')
plt.show()

