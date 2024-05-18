import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

# Load the PLCO data
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv')
y_plco = plco_data['lung']
X_plco = plco_data.drop(columns=['lung'])

# Load NHIS data
# nhis_data = pd.read_csv('Data Files/NHIS_female_30_mean_imputed_data.csv')
nhis_data = pd.read_csv('Data Files/NHIS_female_chosen_mean_imputed_data.csv')

X_nhis = nhis_data.drop(columns=['lung'])
y_nhis = nhis_data['lung']

# Load UK Biobank data
ukb_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv')
X_ukb = ukb_data.drop(columns=['lung'])
y_ukb = ukb_data['lung']

# Determine common features excluding the target
common_features = X_plco.columns.intersection(X_nhis.columns).intersection(X_ukb.columns)

# Apply common features to both datasets
X_plco = X_plco[common_features]
X_nhis = X_nhis[common_features]
X_ukb = X_ukb[common_features]

# Split the PLCO data for training
X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.2, stratify=y_plco)

# Train a Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Testing on UK Biobank data
y_pred_ukb = model.predict_proba(X_ukb)[:, 1]  # Get probability of the positive class
fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
auc_ukb = auc(fpr_ukb, tpr_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label=f'UK Biobank (AUC = {auc_ukb:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('UKB Shallow Female: ROC curve (NB)')
plt.legend(loc='lower right')
plt.show()

# Testing on NHIS data
y_pred_nhis = model.predict_proba(X_nhis)[:, 1]  # Get probability of the positive class
fpr_nhis, tpr_nhis, _ = roc_curve(y_nhis, y_pred_nhis)
auc_nhis = auc(fpr_nhis, tpr_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label=f'NHIS (AUC = {auc_nhis:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('NHIS Shallow Female: ROC curve (NB)')
plt.legend(loc='lower right')
plt.show()





