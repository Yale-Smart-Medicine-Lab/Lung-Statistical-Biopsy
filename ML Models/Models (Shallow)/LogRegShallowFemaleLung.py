import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the PLCO data
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv') 
y_plco = plco_data['lung']
X_plco = plco_data.drop(columns=['lung'])

# Load NHIS data to determine common features
# nhis_data = pd.read_csv('Data Files/female_filtered_70_NHIS_imputed.csv')
# nhis_data = pd.read_csv('Data Files/NHIS_female_30_mean_imputed_data.csv')
nhis_data = pd.read_csv('Data Files/NHIS_female_chosen_mean_imputed_data.csv')


X_nhis = nhis_data.drop(columns=['lung'])
y_nhis = nhis_data['lung']

# Determine common features excluding the target
common_features = X_nhis.columns.intersection(X_plco.columns)

# Apply common features to both datasets
X_plco = X_plco[common_features]
X_nhis = X_nhis[common_features]

# Split the PLCO data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

# Logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation on PLCO test set
accuracy = model.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# Plot ROC Curve for PLCO
y_pred = model.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='lung (AUC = {:.3f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('PLCO Female: ROC curve (Log Reg)')
plt.legend(loc='lower right')
plt.savefig('PLCO_female_lung_ROC_logistic.png', dpi=300, bbox_inches='tight')
plt.show()

# Load UKB data
UKB_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv')
X_ukb = UKB_data[common_features]
y_ukb = UKB_data['lung']

# Testing on UK Biobank Data
y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
roc_auc_ukb = roc_auc_score(y_ukb, y_pred_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label='UK Biobank (AUC = {:.3f})'.format(roc_auc_ukb))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('UKB Female: ROC curve (Log Reg)')
plt.legend(loc='lower right')
plt.savefig('UKB_female_lung_ROC_logistic.png', dpi=300, bbox_inches='tight')
plt.show()

# Testing on NHIS Data
y_pred_nhis = model.predict_proba(X_nhis)[:, 1]
fpr_nhis, tpr_nhis, _ = roc_curve(y_nhis, y_pred_nhis)
roc_auc_nhis = roc_auc_score(y_nhis, y_pred_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label='NHIS (AUC = {:.3f})'.format(roc_auc_nhis))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('NHIS Female: ROC curve (Log Reg)')
plt.legend(loc='lower right')
plt.savefig('NHIS_female_lung_ROC_logistic.png', dpi=300, bbox_inches='tight')
plt.show()

print("Unique values in 'lung' column for NHIS:", np.unique(y_nhis))
