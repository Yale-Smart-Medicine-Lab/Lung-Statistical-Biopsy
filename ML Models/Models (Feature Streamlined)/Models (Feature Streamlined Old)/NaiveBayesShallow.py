import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

def train_and_evaluate_naive_bayes(gender, plco_file, nhis_file, ukb_file, roc_titles, plot_color='blue'):
    # Load datasets based on gender input
    plco_data = pd.read_csv(plco_file.format(gender=gender))
    nhis_data = pd.read_csv(nhis_file)
    ukb_data = pd.read_csv(ukb_file.format(gender=gender))
    
    # Extract target and features, ensuring target 'lung' is not part of the features
    y_plco = plco_data['lung']
    X_plco = plco_data.drop(columns=['lung'])
    y_nhis = nhis_data['lung']
    X_nhis = nhis_data.drop(columns=['lung'])
    y_ukb = ukb_data['lung']
    X_ukb = ukb_data.drop(columns=['lung'])

    # Determine common features between datasets
    common_features = X_plco.columns.intersection(X_nhis.columns).intersection(X_ukb.columns)
    X_plco = X_plco[common_features]
    X_nhis = X_nhis[common_features]
    X_ukb = X_ukb[common_features]

    # Split the PLCO data for training
    X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.2, stratify=y_plco)

    # Train a Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate model and plot ROC curve
    datasets = [(X_ukb, y_ukb, roc_titles[0]), (X_nhis, y_nhis, roc_titles[1])]
    for X, y, title in datasets:
        y_pred = model.predict_proba(X)[:, 1]  # Get probability of the positive class
        fpr, tpr, _ = roc_curve(y, y_pred)
        auc_score = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', color='gray')
        plt.plot(fpr, tpr, color=plot_color, label=f'{title} (AUC = {auc_score:.3f})')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'{title} {gender}: ROC curve (NB)')
        plt.legend(loc='lower right')
        plt.show()

# Example usage
train_and_evaluate_naive_bayes(
    'female',
    'Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv',
    'Data Files/female_filtered_chosen_NHIS_mean_imputed.csv',
    'Data Files/UKB_female_Lung_Imputed_MAIN.csv',
    ['UK Biobank Chosen Features', 'NHIS Chosen Features'],
    plot_color='red'
)

train_and_evaluate_naive_bayes(
    'female',
    'Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv',
    'Data Files/female_filtered_70_NHIS_mean_imputed.csv',
    'Data Files/UKB_female_Lung_Imputed_MAIN.csv',
    ['UK Biobank 70 Features', 'NHIS 70 Features'],
    plot_color='green'
)

train_and_evaluate_naive_bayes(
    'female',
    'Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv',
    'Data Files/female_filtered_30_NHIS_mean_imputed.csv',
    'Data Files/UKB_female_Lung_Imputed_MAIN.csv',
    ['UK Biobank 30 Features', 'NHIS 30 Features'],
    plot_color='blue'
)

train_and_evaluate_naive_bayes(
    'male',
    'Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv',
    'Data Files/male_filtered_chosen_NHIS_mean_imputed.csv',
    'Data Files/UKB_male_Lung_Imputed_MAIN.csv',
    ['UK Biobank Chosen Features', 'NHIS Chosen Features'],
    plot_color='red'
)

train_and_evaluate_naive_bayes(
    'male',
    'Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv',
    'Data Files/male_filtered_70_NHIS_mean_imputed.csv',
    'Data Files/UKB_male_Lung_Imputed_MAIN.csv',
    ['UK Biobank 70 Features', 'NHIS 70 Features'],
    plot_color='green'
)

train_and_evaluate_naive_bayes(
    'male',
    'Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv',
    'Data Files/male_filtered_30_NHIS_mean_imputed.csv',
    'Data Files/UKB_male_Lung_Imputed_MAIN.csv',
    ['UK Biobank 30 Features', 'NHIS 30 Features'],
    plot_color='blue'
)