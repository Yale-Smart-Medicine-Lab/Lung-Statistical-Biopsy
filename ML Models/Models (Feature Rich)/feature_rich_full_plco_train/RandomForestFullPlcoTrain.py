'''
This file contains the code to train and evaluate a Random Forest model on the PLCO and UKB data.
The model is trained on all the PLCO data, validated on the PLCO data, and tested on the UKB data.
Functions: 
    train_evaluate_model: Trains and evaluates the ANN model on the PLCO and UKB data.
        Arguments: plco_data_path, ukb_data_path
    calculate_metrics: Calculates additional metrics for the model.
        Arguments: y_true, y_pred (y_true is the true labels, y_pred is the predicted probabilities of the model)
        Metrics: Precision, F1 Score, Accuracy, Positive Predictive Value (PPV), Negative Predictive Value (NPV), Matthews Correlation Coefficient (MCC), Informedness, Diagnostic Odds Ratio (DOR)
Output: 
    - Prints the metrics for the model on the PLCO training data and the UKB data.
    - Saves the ROC curve plot for the model on the PLCO training data.
    - Saves the ROC curve plot for the model on the UKB data.
    - Saves the metrics for the model on the PLCO training data and the UKB data.
    - Saves the model summary.
    - Saves the model.
'''

from sklearn.ensemble import RandomForestClassifier

# Add the top-level directory to the sys.path
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from helper_functions import print_metrics, plot_auc, train_evaluate_model

# Paths to male and female datasets
male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_plco_train_metrics, male_ukb_metrics = train_evaluate_model(male_plco_path, male_ukb_path, RandomForestClassifier, 'Random Forest (Male): Training Loss', 'ML Models/Models (Feature Rich)/feature_rich_full_plco_train/feature_rich_full_train_photos/RandomForestFeatureRichFullTrainMaleLoss.png')

(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_plco_train_metrics, female_ukb_metrics = train_evaluate_model(female_plco_path, female_ukb_path, RandomForestClassifier, 'Random Forest (Female): Training Loss', 'ML Models/Models (Feature Rich)/feature_rich_full_plco_train/feature_rich_full_train_photos/RandomForestFeatureRichFullTrainFemaleLoss.png')

print_metrics(male_plco_train_metrics, male_ukb_metrics,
              female_plco_train_metrics, female_ukb_metrics)

plot_auc(male_fpr_plco, male_tpr_plco, male_auc_plco,
             male_fpr_ukb, male_tpr_ukb, male_auc_ukb,
             female_fpr_plco, female_tpr_plco, female_auc_plco,
             female_fpr_ukb, female_tpr_ukb, female_auc_ukb,
             filename='ML Models/Models (Feature Rich)/feature_rich_full_plco_train/feature_rich_full_train_photos/RandomForestFeatureRichFullTrain.png',
             title='Random Forest: ROC Curves for Lung Cancer Prediction (FT)')
