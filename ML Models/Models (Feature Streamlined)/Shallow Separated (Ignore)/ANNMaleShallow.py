import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(plco_file, nhis_file, ukb_file, roc_titles, plot_color='blue'):
    # Load datasets
    plco_data = pd.read_csv(plco_file)
    nhis_data = pd.read_csv(nhis_file)
    ukb_data = pd.read_csv(ukb_file)

    # Ensure 'lung' is the target and not part of the features
    if 'lung' in plco_data.columns:
        y_plco = plco_data['lung']
        X_plco = plco_data.drop(columns=['lung'])
    else:
        raise ValueError("Target variable 'lung' is not in PLCO dataframe.")

    if 'lung' in nhis_data.columns:
        y_nhis = nhis_data['lung']
        X_nhis = nhis_data.drop(columns=['lung'])
    else:
        raise ValueError("Target variable 'lung' is not in NHIS dataframe.")

    if 'lung' in ukb_data.columns:
        y_ukb = ukb_data['lung']
        X_ukb = ukb_data.drop(columns=['lung'])
    else:
        raise ValueError("Target variable 'lung' is not in UKB dataframe.")

    # Determine common features between datasets
    common_features = X_nhis.columns.intersection(X_plco.columns).intersection(X_ukb.columns)
    X_plco = X_plco[common_features]
    X_nhis = X_nhis[common_features]
    X_ukb = X_ukb[common_features]

    # Split PLCO data into training and testing sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

    # Define and compile the model
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.AUC(from_logits=False),
                 tf.keras.metrics.Precision(name='PPV'),
                 tf.keras.metrics.Recall(name='Sensitivity')]
    )

    # Train the model
    model.fit(X_train, y_train, batch_size=1024, epochs=10, shuffle=True, verbose=2)

    # Predict and plot ROC curve for UKB and NHIS data
    datasets = [(X_ukb, y_ukb, 'ROC_UKBiobank.png', roc_titles[0]), 
                (X_nhis, y_nhis, 'ROC_NHIS.png', roc_titles[1])]
    
    for X, y, filename, title in datasets:
        y_pred = model.predict(X).ravel()
        fpr, tpr, _ = roc_curve(y, y_pred)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', color='gray')
        plt.plot(fpr, tpr, color=plot_color, label=f'{title} (AUC = {auc(fpr, tpr):.3f})')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'Male ANN: ROC curve for {title}')
        plt.legend(loc='lower right')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
train_and_evaluate_model('Data Files/PLCO_Male_Lung_Data_MAIN_imputed.csv',
                         'Data Files/male_filtered_chosen_NHIS_mean_imputed.csv',
                         'Data Files/UKB_Male_Lung_Imputed_MAIN.csv',
                         ['UK Biobank (Chosen Variables)', 'NHIS (Chosen Variables)'],
                         plot_color='red')

train_and_evaluate_model('Data Files/PLCO_Male_Lung_Data_MAIN_imputed.csv',
                         'Data Files/male_filtered_70_NHIS_mean_imputed.csv',
                         'Data Files/UKB_Male_Lung_Imputed_MAIN.csv',
                         ['UK Biobank (70 Feature)', 'NHIS (70 Feature)'],
                         plot_color='green')

train_and_evaluate_model('Data Files/PLCO_Male_Lung_Data_MAIN_imputed.csv',
                         'Data Files/male_filtered_30_NHIS_mean_imputed.csv',
                         'Data Files/UKB_Male_Lung_Imputed_MAIN.csv',
                         ['UK Biobank (30 Feature)', 'NHIS (30 Feature)'],
                         plot_color='blue')



