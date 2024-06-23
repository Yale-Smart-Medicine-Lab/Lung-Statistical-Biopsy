import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/PLCO_male_Lung_Data_MAIN_imputed.csv')

# Separate features and target
X = data.drop(columns=['lung'])
y = data['lung']

# Split data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Model
model = GaussianNB()

# Training
model.fit(X_train, y_train)

# Evaluation
accuracy = model.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# Plot ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='lung (AUC = {:.3f})'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate (Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('PLCO Male: ROC curve (Naive Bayes)')
plt.legend(loc='lower right')
plt.savefig('male_lung_ROC_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()

# Load UKB data
ukb_data = pd.read_csv('/Users/teresanguyen/Documents/Lung-Statistical-Biopsy/Data Files/UKB_male_Lung_Imputed_MAIN.csv')

# Assuming 'lung' column contains the target variable
x_ukb = ukb_data[X.columns]  # Assuming 'X' contains the same features as 'X_train'

# Predict using the trained model
y_pred_ukb_proba = model.predict_proba(x_ukb)[:, 1]

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
fpr_ukb, tpr_ukb, thresholds_ukb = roc_curve(ukb_data['lung'], y_pred_ukb_proba)
plt.plot(fpr_ukb, tpr_ukb, label='lung (AUC = {:.3f})'.format(auc(fpr_ukb, tpr_ukb)))
plt.xlabel('False positive rate (Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('Male Lung Cancer UK Biobank Testing: ROC curve (Naive Bayes)')
plt.legend(loc='lower right')
plt.savefig('maleCancerROCUKBiobank_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.show()
