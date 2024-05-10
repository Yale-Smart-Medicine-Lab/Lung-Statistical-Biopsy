import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Load PLCO data
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv')

# Load NHIS data to determine common features
nhis_data = pd.read_csv('Data Files/female_filtered_70_NHIS_imputed.csv')
common_features = nhis_data.columns.intersection(plco_data.columns)

# Keep only the common features and the target in PLCO data
X_plco = plco_data[common_features]
y_plco = plco_data['lung']

# Split PLCO data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

# Model
model = keras.models.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]],
                       kernel_initializer=keras.initializers.glorot_normal(),
                       bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(80, activation='relu',
                       kernel_initializer=keras.initializers.glorot_normal(),
                       bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(1, activation='sigmoid')
])

# Loss and optimizer
loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1)
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.AUC(from_logits=False),
           tf.keras.metrics.Precision(name='PPV'),
           tf.keras.metrics.Recall(name='Sensitivity')]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# Training
batch_size = 1024
epochs = 10
model.fit(X_train.to_numpy(), y_train.to_numpy(), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# Load UK Biobank data and limit to common features
ukb_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv')
X_ukb = ukb_data[common_features]
y_ukb = ukb_data['lung']

# Testing on UK Biobank
y_pred_ukb = model.predict(X_ukb)
fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label='UK Biobank (AUC = {:.3f})'.format(auc(fpr_ukb, tpr_ukb)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Female ANN: ROC curve for UK Biobank')
plt.legend(loc='lower right')
plt.savefig('ROC_UKBiobank.png', dpi=300, bbox_inches='tight')
plt.show()

# Load NHIS data and limit to common features
X_nhis = nhis_data[common_features]
y_nhis = nhis_data['lung']

# Ensure y_nhis contains only two unique values (binary)
print("Unique values in 'lung' column:", np.unique(y_nhis))
y_pred_nhis = model.predict(X_nhis).ravel()
fpr_nhis, tpr_nhis, _ = roc_curve(y_nhis, y_pred_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label='NHIS (AUC = {:.3f})'.format(auc(fpr_nhis, tpr_nhis)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Female ANN: ROC curve for NHIS dataset')
plt.legend(loc='lower right')
plt.savefig('ROC_NHIS.png', dpi=300, bbox_inches='tight')
plt.show()
