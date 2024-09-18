
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from helper_functions import ann_cross_validation_and_train, ann_plot_loss_curves, table_of_metrics, plot_roc_curves

male_plco_path = 'data_files/imputed/PLCO_male_Lung_Data_MAIN_imputed.csv'
male_ukb_path = 'data_files/imputed/UKB_male_Lung_Imputed_MAIN.csv'
female_plco_path = 'data_files/imputed/PLCO_female_Lung_Data_MAIN_imputed.csv'
female_ukb_path = 'data_files/imputed/UKB_female_Lung_Imputed_MAIN.csv'

(male_fpr_plco, male_tpr_plco, male_auc_plco), (male_fpr_ukb, male_tpr_ukb, male_auc_ukb), male_cv_means, male_cv_stds, male_plco_train_metrics, male_ukb_metrics, male_history_list, male_history = ann_cross_validation_and_train(male_plco_path, male_ukb_path)
(female_fpr_plco, female_tpr_plco, female_auc_plco), (female_fpr_ukb, female_tpr_ukb, female_auc_ukb), female_cv_means, female_cv_stds, female_plco_train_metrics, female_ukb_metrics, female_history_list, female_history = ann_cross_validation_and_train(female_plco_path, female_ukb_path)

ann_plot_loss_curves(male_history_list, 'ANN Feature Rich CV: Male Loss Curves')
ann_plot_loss_curves(female_history_list, 'ANN Feature Rich CV: Female Loss Curves')

table_of_metrics('ANN Male Cross-Validation Metrics (Mean ± Std)', male_cv_means, male_cv_stds)
table_of_metrics('ANN Male Training Metrics', None, None, male_plco_train_metrics)
table_of_metrics('ANN Male Testing Metrics', None, None, male_ukb_metrics)

table_of_metrics('ANN Female Cross-Validation Metrics (Mean ± Std)', female_cv_means, female_cv_stds)
table_of_metrics('ANN Female Training Metrics', None, None, female_plco_train_metrics)
table_of_metrics('ANN Female Testing Metrics', None, None, female_ukb_metrics)

plot_roc_curves(male_fpr_plco, male_tpr_plco, male_fpr_ukb, male_tpr_ukb, female_fpr_plco, female_tpr_plco, female_fpr_ukb, female_tpr_ukb, male_auc_plco, male_auc_ukb, female_auc_plco, female_auc_ukb, 'ANN Feature Rich: ROC Curves for Lung Cancer Prediction CV')
