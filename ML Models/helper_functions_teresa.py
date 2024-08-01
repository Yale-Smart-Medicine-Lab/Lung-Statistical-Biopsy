import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix, log_loss

# This function calculates the following metrics: precision, f1, accuracy, ppv, npv, mcc, informedness, dor
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    
    precision = precision_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    accuracy = accuracy_score(y_true, y_pred_labels)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred_labels)
    informedness = (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0
    dor = (tp / fn) / (fp / tn) if fn > 0 and fp > 0 and tn > 0 else 0

    return precision, f1, accuracy, ppv, npv, mcc, informedness, dor

# Artificial Neural Network model
def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[input_shape],
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(80, activation='relu',
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(1)
        ])
    
    model.compile(
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),]
    )

    return model 

# all the cross validation models
def ann_cross_validation_and_train(plco_data_path, ukb_data_path): 
    # Load in the PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []
    loss_curves = []

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train.iloc[train_index], X_plco_train.iloc[val_index]
        y_train, y_val = y_plco_train.iloc[train_index], y_plco_train.iloc[val_index]
        model = create_model(X_train.shape[1])
        loss = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=1024, epochs=10, shuffle=True, verbose=0)
        y_pred_val = model.predict(X_val)
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))
        loss_curves.append(loss)
    
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)

    model = create_model(X_plco_train.shape[1])
    model.fit(X_plco_train, y_plco_train, batch_size=1024, epochs=10, shuffle=True, verbose=2, validation_split=0.1)

    model.evaluate(X_plco_train, y_plco_train, verbose=2, batch_size=1024)
    y_pred_plco_train = model.predict(X_plco_train)

    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)

    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']

    y_pred_ukb = model.predict(X_ukb)
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), cv_means, cv_stds, plco_train_metrics, ukb_metrics, loss_curves, loss

def log_reg_cross_validation_and_train(plco_data_path, ukb_data_path, solver, max_iter):
    plco_data = pd.read_csv(plco_data_path)
    ukb_data = pd.read_csv(ukb_data_path)
    common_features = list(set(plco_data.columns) & set(ukb_data.columns))
    common_features.remove('lung')

    X_plco_train = plco_data[common_features].values
    y_plco_train = plco_data['lung'].values

    scaler = StandardScaler()
    X_plco_train = scaler.fit_transform(X_plco_train)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []
    loss_curves = []

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train[train_index], X_plco_train[val_index]
        y_train, y_val = y_plco_train[train_index], y_plco_train[val_index]
        model = LogisticRegression(solver=solver, max_iter=max_iter, verbose=1)
        model.fit(X_train, y_train)
        y_pred_val = model.predict_proba(X_val)[:, 1]
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))

        losses = []
        for i in range(1, model.n_iter_[0] + 1):
            model_iter = LogisticRegression(max_iter=i, solver=solver, warm_start=True)
            model_iter.fit(X_train, y_train)
            y_pred_iter = model_iter.predict_proba(X_val)[:, 1]
            losses.append(log_loss(y_val, y_pred_iter))
        loss_curves.append(losses)
    
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)

    model.fit(X_plco_train, y_plco_train)

    y_pred_plco_train = model.predict_proba(X_plco_train)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    X_ukb = ukb_data[common_features].values
    X_ukb = scaler.transform(X_ukb)
    y_ukb = ukb_data['lung'].values

    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    coefficients = model.coef_[0]
    df_coefficients = pd.DataFrame({'Feature': common_features, 'Coefficient': coefficients})

    if 'female' in plco_data_path.lower(): 
        df_coefficients.to_csv('ML Models/coefficients/' + 'female coefficients' + '.csv', index=False)
    else: 
        df_coefficients.to_csv('ML Models/coefficients/' + 'male coefficients' + '.csv', index=False)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), cv_means, cv_stds, plco_train_metrics, ukb_metrics, loss_curves


def ann_plot_loss_curves(history_list, title):
    plt.figure(figsize=(10, 5))
    for history in history_list:
        plt.plot(history.history['loss'], 'b-', alpha=0.3)
        plt.plot(history.history['val_loss'], 'r-', alpha=0.3)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('ML Models/models/' + title + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def log_reg_plot_loss_curves(male_history_list, female_history_list, title):
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(male_history_list):
        plt.plot(losses, label=f'Fold {i + 1}', alpha=0.3)
    for i, losses in enumerate(female_history_list):
        plt.plot(losses, label=f'Fold {i + 1}', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.5)
    plt.savefig('ML Models/models/' + title + '.png' , dpi=300, bbox_inches='tight')
    plt.show()

def table_of_metrics(title, cv_means=None, cv_stds=None, metrics_values=None):
    metrics = ["Precision", "F1 Score", "Accuracy", "PPV", "NPV", "MCC", "Informedness", "DOR"]
    data = []

    # Checking if the title indicates it's for cross-validation metrics
    if "Cross-Validation" in title:
        if cv_means is None or cv_stds is None:
            raise ValueError("cv_means and cv_stds must be provided for cross-validation metrics.")
        data.append([title] + ["{:.4f} ± {:.4f}".format(mean, std) for mean, std in zip(cv_means, cv_stds)])
    else:
        if metrics_values is None:
            raise ValueError("metrics_values must be provided for non-cross-validation metrics.")
        data.append([title] + ["{:.4f}".format(metric) for metric in metrics_values])
    
    fig, ax = plt.subplots(figsize=(12, 1.5))  # Adjusted for potentially better fitting
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=["Title"] + metrics, cellLoc='center', loc='center', colColours=["#D3D3D3"] * (len(metrics) + 1))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)  # Adjust scale if necessary

    # Set the background color for the top row
    for (i, key), cell in table.get_celld().items():
        if i == 0:  # Only the top row
            cell.set_facecolor('#D3D3D3')
            cell.set_edgecolor('black')
            cell.set_height(0.1)  # Adjust the height if necessary

    plt.title(title, fontsize=12, weight='bold')
    plt.savefig(f'ML Models/metrics/{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(male_fpr_plco, male_tpr_plco, male_fpr_ukb, male_tpr_ukb, female_fpr_plco, female_tpr_plco, female_fpr_ukb, female_tpr_ukb, male_auc_plco, male_auc_ukb, female_auc_plco, female_auc_ukb, title):  
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
    plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
    plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
    plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Neural Network: ROC Curves for Lung Cancer Prediction CV')
    plt.legend(loc='lower right')
    plt.savefig('ML Models/models/' + title + '.png', dpi=300, bbox_inches='tight')
    plt.show()






