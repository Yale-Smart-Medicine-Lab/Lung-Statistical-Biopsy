import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Function to load data and extract common features
def load_and_extract_features(file_paths):
    datasets = [pd.read_csv(path) for path in file_paths]
    common_features = set(datasets[0].columns)
    for dataset in datasets[1:]:
        common_features.intersection_update(dataset.columns)
    common_features.remove('lung')  # Ensure 'lung' is not part of the features
    return [dataset[list(common_features) + ['lung']] for dataset in datasets], list(common_features)

# Function to prepare data, train model, and return the model
def train_model(X, y):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Define the neural network model
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, batch_size=1024, epochs=10, verbose=2)
    return model

# Function to evaluate the model and plot ROC curves
def evaluate_and_plot(model, common_features, data_paths, labels):
    plt.figure(figsize=(12, 6))

    for i, (data_path, label) in enumerate(zip(data_paths, labels)):
        # Load and subset data based on common features
        data = pd.read_csv(data_path)
        y_test = data['lung']
        X_test = data[common_features]  # Ensure only trained features are used

        y_pred = model.predict(X_test).ravel()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, len(data_paths), i+1)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {label}')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# Paths to datasets
plco_path_female = 'Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv'
ukb_path_female = 'Data Files/UKB_Female_Lung_Imputed_MAIN.csv'
nhis_path_female = 'Data Files/NHIS_female_chosen_mean_imputed_data.csv'

# Load datasets and extract common features
datasets_female, common_features_female = load_and_extract_features([plco_path_female, ukb_path_female, nhis_path_female])
plco_data_female = datasets_female[0]

# Train model on PLCO data
model = train_model(plco_data_female.drop(columns=['lung']), plco_data_female['lung'])

# Evaluate on UK Biobank and NHIS data
evaluate_and_plot(model, common_features_female, [ukb_path_female, nhis_path_female], ['UKB Female', 'NHIS Female'])

plco_path_male = 'Data Files/PLCO_Male_Lung_Data_MAIN_imputed.csv'
ukb_path_male = 'Data Files/UKB_Male_Lung_Imputed_MAIN.csv'
nhis_path_male = 'Data Files/NHIS_male_chosen_mean_imputed_data.csv'

datasets_male, common_features_male = load_and_extract_features([plco_path_male, ukb_path_male, nhis_path_male])





