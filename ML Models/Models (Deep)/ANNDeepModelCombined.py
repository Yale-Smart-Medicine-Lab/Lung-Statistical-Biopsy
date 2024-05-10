import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

def prepare_data(file_path, X_columns):
    data = pd.read_csv(file_path)
    X = data[X_columns]  # Make sure to use the same columns as PLCO models
    y = data['lung']  # Ensure this column is binary (0 or 1)
    return X, y

def build_and_train_model(X_train, y_train):
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    model.fit(X_train.to_numpy(), y_train.to_numpy(), batch_size=1024, epochs=10, verbose=0)
    return model

def plot_roc_curves(model, X_test, y_test, label, ax):
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    ax.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})')

# Load and prepare PLCO data
X_train_female, X_test_female, y_train_female, y_test_female = prepare_data('Female_Lung_Data_Greg_imputed.csv')
X_train_male, X_test_male, y_train_male, y_test_male = prepare_data('Male_Lung_Data_Greg_imputed.csv')

# Build and train models for PLCO
model_female = build_and_train_model(X_train_female, y_train_female)
model_male = build_and_train_model(X_train_male, y_train_male)

# Load UK Biobank data
X_ukb_female, y_ukb_female = prepare_data('imputedFemaleDataGreg2.csv', X_train_female.columns)
X_ukb_male, y_ukb_male = prepare_data('imputedMaleDataGreg2.csv', X_train_male.columns)

# Plot ROC curves for PLCO
fig, ax = plt.subplots(figsize=(6, 6))
plot_roc_curves(model_female, X_test_female, y_test_female, 'Female PLCO', ax)
plot_roc_curves(model_male, X_test_male, y_test_male, 'Male PLCO', ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False positive rate (Specificity)')
ax.set_ylabel('True positive rate (Sensitivity)')
ax.set_title('Neural Network: ROC curve comparison for PLCO')
ax.legend(loc='lower right')
plt.show()

# Plot ROC curves for UKB using PLCO models
fig, ax = plt.subplots(figsize=(6, 6))
plot_roc_curves(model_female, X_ukb_female, y_ukb_female, 'Female UKB', ax)
plot_roc_curves(model_male, X_ukb_male, y_ukb_male, 'Male UKB', ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False positive rate (Specificity)')
ax.set_ylabel('True positive rate (Sensitivity)')
ax.set_title('Neural Network: ROC curve comparison for UK Biobank')
ax.legend(loc='lower right')
plt.show()
