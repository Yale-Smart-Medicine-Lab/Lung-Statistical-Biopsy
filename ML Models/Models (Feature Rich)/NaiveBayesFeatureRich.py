import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

# Define a function to prepare data, train a model, and calculate ROC metrics
def process_and_predict(data_path, columns=None):
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['lung'])
    y = data['lung']

    if columns is not None:
        X = X[columns]  # Use the same columns used in training

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Create and train the GaussianNB model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score, model, X_train.columns

# Process each dataset
fpr_plco_f, tpr_plco_f, auc_plco_f, model_f, columns_f = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_female_Lung_Data_MAIN_imputed.csv')
fpr_plco_m, tpr_plco_m, auc_plco_m, model_m, columns_m = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv')

# Process UKB data using trained models and same features
fpr_ukb_f, tpr_ukb_f, auc_ukb_f, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_female_Lung_Imputed_MAIN.csv', columns_f)
fpr_ukb_m, tpr_ukb_m, auc_ukb_m, _, _ = process_and_predict('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv', columns_m)

# Plot all ROC curves
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.plot(fpr_plco_f, tpr_plco_f, label=f'PLCO Female (AUC = {auc_plco_f:.3f})', color='red')
plt.plot(fpr_plco_m, tpr_plco_m, label=f'PLCO Male (AUC = {auc_plco_m:.3f})', color='blue')
plt.plot(fpr_ukb_f, tpr_ukb_f, label=f'UKB Female (AUC = {auc_ukb_f:.3f})', color='green')
plt.plot(fpr_ukb_m, tpr_ukb_m, label=f'UKB Male (AUC = {auc_ukb_m:.3f})', color='orange')
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Senstivity)')
plt.title('Naive Bayes: ROC Curves for Lung Cancer Prediction')
plt.legend(loc='lower right')
plt.savefig('ML Models/Models (Deep)/Feature Rich Photos/NaiveBayesFeatureRich.png', dpi=300, bbox_inches='tight')
plt.show()
