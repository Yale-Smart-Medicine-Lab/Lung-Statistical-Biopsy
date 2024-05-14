import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt

# Load the datasets
plco_data = pd.read_csv('Data Files/PLCO_Female_Lung_Data_MAIN_imputed.csv')
nhis_data = pd.read_csv('Data Files/female_filtered_70_NHIS_imputed.csv').drop_duplicates()
ukb_data = pd.read_csv('Data Files/UKB_Female_Lung_Imputed_MAIN.csv').drop_duplicates()

# Ensure "lung" is treated as the target and removed from the feature set
target_variable = 'lung'
if target_variable in plco_data.columns:
    y_plco = plco_data[target_variable]
    X_plco = plco_data.drop(columns=[target_variable])
else:
    raise ValueError(f"The target variable '{target_variable}' is not in the dataframe.")

# Define common features based on available features excluding the target
common_features = nhis_data.columns.intersection(X_plco.columns)

# Filter the features in the dataset
X_plco = X_plco[common_features]

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True)
auc_scores = []

# Loop over each fold
for train_index, test_index in kfold.split(X_plco, y_plco):
    X_train, X_test = X_plco.iloc[train_index], X_plco.iloc[test_index]
    y_train, y_test = y_plco.iloc[train_index], y_plco.iloc[test_index]

    # Model definition with an explicit input layer
    model = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Fit model
    model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=10, batch_size=1024, verbose=0)

    # Evaluate model on the test set
    y_pred = model.predict(X_test).ravel()
    auc_score = sk_metrics.roc_auc_score(y_test, y_pred)
    auc_scores.append(auc_score)

print(f"Cross-validated AUC scores: {auc_scores}")
print(f"Mean AUC: {np.mean(auc_scores)}, Standard Deviation: {np.std(auc_scores)}")

# Adjust the feature selection for UK Biobank and NHIS datasets
X_ukb = ukb_data[common_features]
y_ukb = ukb_data[target_variable]
X_nhis = nhis_data[common_features]
y_nhis = nhis_data[target_variable]

# Predictions and plotting for UK Biobank
y_pred_ukb = model.predict(X_ukb).ravel()
fpr_ukb, tpr_ukb, _ = sk_metrics.roc_curve(y_ukb, y_pred_ukb)
auc_ukb = sk_metrics.auc(fpr_ukb, tpr_ukb)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ukb, tpr_ukb, label=f'UK Biobank (AUC = {auc_ukb:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for UK Biobank')
plt.legend(loc='lower right')
plt.show()

# Predictions and plotting for NHIS
y_pred_nhis = model.predict(X_nhis).ravel()
fpr_nhis, tpr_nhis, _ = sk_metrics.roc_curve(y_nhis, y_pred_nhis)
auc_nhis = sk_metrics.auc(fpr_nhis, tpr_nhis)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nhis, tpr_nhis, label=f'NHIS (AUC = {auc_nhis:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for NHIS')
plt.legend(loc='lower right')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Calculate AUC
log_reg_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Logistic Regression AUC: {log_reg_auc}")

# Feature importance
feature_importance = pd.Series(log_reg.coef_[0], index=X_train.columns)
print("Feature importances:\n", feature_importance.sort_values(ascending=False))



