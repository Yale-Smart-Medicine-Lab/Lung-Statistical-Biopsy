import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def random_forest_analysis(plco_basepath, ukb_basepath, nhis_path, titleplot, gender, plot_color='blue', n_estimators=100, max_depth=None):
    # Construct file paths based on the provided gender
    plco_file = f'{plco_basepath}_{gender}_Lung_Data_MAIN_imputed.csv'
    nhis_file = nhis_path
    ukb_file = f'{ukb_basepath}_{gender}_Lung_Imputed_MAIN.csv'

    # Load and preprocess datasets
    def load_and_preprocess(file):
        data = pd.read_csv(file)
        y = data['lung']
        X = data.drop(columns=['lung'])
        return X, y

    X_plco, y_plco = load_and_preprocess(plco_file)
    X_nhis, y_nhis = load_and_preprocess(nhis_file)
    X_ukb, y_ukb = load_and_preprocess(ukb_file)

    # Determine common features
    common_features = X_nhis.columns.intersection(X_plco.columns).intersection(X_ukb.columns)

    # Apply common features to all datasets
    X_plco = X_plco[common_features]
    X_nhis = X_nhis[common_features]
    X_ukb = X_ukb[common_features]

    # Split PLCO data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_plco, y_plco, test_size=0.3, stratify=y_plco)

    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Function to plot ROC curves
    def plot_roc_curve(y_true, y_scores, data, title, gender, filename):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, color=plot_color, label=f'{title} (AUC = {roc_auc:.3f})')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'{data} {title} {gender}: ROC curve (RF)')
        plt.legend(loc='lower right')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    # Evaluate on PLCO test set and plot ROC
    y_pred_plco = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_pred_plco, 'PLCO', titleplot, gender, 'PLCO_lung_ROC_random_forest.png')

    # Evaluate on UKB and plot ROC
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    plot_roc_curve(y_ukb, y_pred_ukb, 'UKB', titleplot, gender, 'UKB_lung_ROC_random_forest.png')

    # Evaluate on NHIS and plot ROC
    y_pred_nhis = model.predict_proba(X_nhis)[:, 1]
    plot_roc_curve(y_nhis, y_pred_nhis, 'NHIS', titleplot, gender, 'NHIS_lung_ROC_random_forest.png')

# Example usage of the function:
random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/female_filtered_chosen_NHIS_mean_imputed.csv',
    titleplot='Chosen Features',
    gender='female',  
    plot_color='red',  
    n_estimators=200, 
    max_depth=10  
)

random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/female_filtered_70_NHIS_mean_imputed.csv',
    titleplot='70 Features',
    gender='female',  
    plot_color='green',  
    n_estimators=200, 
    max_depth=10  
)

random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/female_filtered_30_NHIS_mean_imputed.csv',
    titleplot='30 Features',
    gender='female',  
    plot_color='blue',  
    n_estimators=200, 
    max_depth=10  
)

random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/male_filtered_chosen_NHIS_mean_imputed.csv',
    titleplot='Chosen Features',
    gender='male',  
    plot_color='red',  
    n_estimators=200, 
    max_depth=10  
)

random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/male_filtered_70_NHIS_mean_imputed.csv',
    titleplot='70 Features',
    gender='male',  
    plot_color='green',  
    n_estimators=200, 
    max_depth=10  
)

random_forest_analysis(
    plco_basepath='Data Files/PLCO',
    ukb_basepath='Data Files/UKB',
    nhis_path='Data Files/male_filtered_30_NHIS_mean_imputed.csv',
    titleplot='30 Features',
    gender='male',  
    plot_color='blue',  
    n_estimators=200, 
    max_depth=10  
)