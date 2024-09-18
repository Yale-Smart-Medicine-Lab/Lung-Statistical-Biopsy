import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from helper_functions import nb_cross_validation_and_train, plot_loss_curves, table_of_metrics, plot_roc_curves

male_plco_path = 'Input/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'Input/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'Input/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'Input/UKB_female_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_cv_means, male_cv_stds, male_plco_train_metrics, male_ukb_metrics, male_loss_curves = nb_cross_validation_and_train(male_plco_path, male_ukb_path)
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_cv_means, female_cv_stds, female_plco_train_metrics, female_ukb_metrics, female_loss_curves = nb_cross_validation_and_train(female_plco_path, female_ukb_path)

plot_loss_curves(male_loss_curves, female_loss_curves, 'Naive Bayes Feature Rich CV: Male and Female Loss Curves')

table_of_metrics('Naive Bayes Male Cross-Validation Metrics (Mean ± Std)', male_cv_means, male_cv_stds)
table_of_metrics('Naive Bayes Male Training Metrics', None, None, male_plco_train_metrics)
table_of_metrics('Naive Bayes Male Testing Metrics', None, None, male_ukb_metrics)

table_of_metrics('Naive Bayes Female Cross-Validation Metrics (Mean ± Std)', female_cv_means, female_cv_stds)
table_of_metrics('Naive Bayes Female Training Metrics', None, None, female_plco_train_metrics)
table_of_metrics('Naive Bayes Female Testing Metrics', None, None, female_ukb_metrics)

plot_roc_curves(male_fpr_plco, male_tpr_plco, male_fpr_ukb, male_tpr_ukb, female_fpr_plco, female_tpr_plco, female_fpr_ukb, female_tpr_ukb, male_auc_plco, male_auc_ukb, female_auc_plco, female_auc_ukb, 'Naive Bayes Feature Rich: ROC Curves for Lung Cancer Prediction CV')