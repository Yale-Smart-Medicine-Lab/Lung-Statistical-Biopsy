import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Load PLCO data
plco_data = pd.read_csv('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv')
X_plco = plco_data.drop(columns=['lung'])
y_plco = plco_data['lung']

# Split PLCO data into training and testing sets with stratified sampling
X_train_plco, X_test_plco, y_train_plco, y_test_plco = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

# Define the model
model = keras.models.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=[X_train_plco.shape[1]]),
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
model.fit(X_train_plco, y_train_plco, epochs=10, batch_size=1024, verbose=2)

# Evaluate on PLCO test set and calculate ROC
y_pred_plco = model.predict(X_test_plco)
fpr_plco, tpr_plco, thresholds_plco = roc_curve(y_test_plco, y_pred_plco)
auc_plco = auc(fpr_plco, tpr_plco)

# Load UKB data
ukb_data = pd.read_csv('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv')
X_ukb = ukb_data[X_train_plco.columns]
y_ukb = ukb_data['lung']

# Predict on UKB data and calculate ROC
y_pred_ukb = model.predict(X_ukb)
fpr_ukb, tpr_ukb, thresholds_ukb = roc_curve(y_ukb, y_pred_ukb)
auc_ukb = auc(fpr_ukb, tpr_ukb)

# Plot both ROC curves
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(fpr_plco, tpr_plco, label=f'PLCO (AUC = {auc_plco:.3f})')
plt.plot(fpr_ukb, tpr_ukb, label=f'UKB (AUC = {auc_ukb:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network: ROC Curves for Female Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.show()
