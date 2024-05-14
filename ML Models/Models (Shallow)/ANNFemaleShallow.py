import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Load PLCO data and separate target
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv')
y_plco = plco_data['lung']
X_plco = plco_data.drop(columns=['lung'])
print(f"Number of entries in PLCO data: {plco_data.shape[0]}")

# Load NHIS data to determine common features, separate target
nhis_data = pd.read_csv('Data Files/female_filtered_70_NHIS_imputed.csv')
y_nhis = nhis_data['lung']
X_nhis = nhis_data.drop(columns=['lung'])
common_features = X_nhis.columns.intersection(X_plco.columns)

# Apply common features to both datasets
X_plco = X_plco[common_features]
X_nhis = X_nhis[common_features]

# Split PLCO data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

# Define the neural network model
model = keras.models.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Set loss, optimizer, and metrics
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
             tf.keras.metrics.AUC(from_logits=False),
             tf.keras.metrics.Precision(name='PPV'),
             tf.keras.metrics.Recall(name='Sensitivity')]
)

# Train the model
model.fit(X_train, y_train, batch_size=1024, epochs=10, verbose=2)

# Load and prepare UK Biobank data
ukb_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv')
y_ukb = ukb_data['lung']
X_ukb = ukb_data.drop(columns=['lung'])[common_features]
print(f"Number of entries in UK Biobank data: {ukb_data.shape[0]}")

# Test the model on UK Biobank data
y_pred_ukb = model.predict(X_ukb)
fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label=f'UK Biobank (AUC = {auc(fpr_ukb, tpr_ukb):.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ANN Female: ROC Curve for UK Biobank')
plt.legend(loc='lower right')
plt.savefig('ROC_UKBiobank.png', dpi=300, bbox_inches='tight')
plt.show()

# Test the model on NHIS data
y_pred_nhis = model.predict(X_nhis).ravel()
fpr_nhis, tpr_nhis, _ = roc_curve(y_nhis, y_pred_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label=f'NHIS (AUC = {auc(fpr_nhis, tpr_nhis):.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for NHIS')
plt.legend(loc='lower right')
plt.savefig('ROC_NHIS.png', dpi=300, bbox_inches='tight')
plt.show()
