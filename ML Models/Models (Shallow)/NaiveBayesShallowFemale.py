import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

# Load in the data 
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv')
nhis_data = pd.read_csv('Data Files/female_filtered_70_NHIS_imputed.csv')
ukb_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv')

# Find the common featurs intersection between plco and NHIS 
common_features = nhis_data.columns.intersection(plco_data.columns)

# use only the features present in NHIS
X_plco = plco_data[common_features]
y_plco = plco_data['lung']

# split the plco data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

# Naive Bayes model
model = GaussianNB()

# Training
model.fit(X_train, y_train)

# Evaluation
accuracy = model.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# Plot ROC Curve
y_pred = model.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='lung (AUC = {:.3f})'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate (Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('PLCO Shallow Female: ROC curve (NB)')
plt.legend(loc='lower right')
plt.savefig('PLCO_female_lung_ROC_NB_shallow.png', dpi=300, bbox_inches='tight')
plt.show()

# UK Biobank data
X_ukb = ukb_data[common_features]
y_ukb = ukb_data['lung']

# Testing on UK Biobank Data
y_pred_ukb = model.predict(X_ukb)
fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label='UK Biobank (AUC = {:.3f})'.format(auc(fpr_ukb, tpr_ukb)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('UKB Shallow Female: ROC curve (NB)')
plt.legend(loc='lower right')
plt.savefig('UKB_female_lung_ROC_NB_shallow.png', dpi=300, bbox_inches='tight')
plt.show()

# NHIS data
X_nhis = nhis_data[common_features]
y_nhis = nhis_data['lung']
y_pred_nhis = model.predict(X_nhis)
fpr_nhis, tpr_nhis, _ = roc_curve(y_nhis, y_pred_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label='NHIS (AUC = {:.3f})'.format(auc(fpr_nhis, tpr_nhis)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('NHIS Shallow Female: ROC curve (NB)')
plt.legend(loc='lower right')
plt.savefig('NHIS_female_lung_ROC_NB_shallow.png', dpi=300, bbox_inches='tight')
plt.show()




