
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
        df_coefficients.to_csv('coefficients/female_coefficients.csv', index=False)
    else: 
        df_coefficients.to_csv('coefficients/male_coefficients.csv', index=False)

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), cv_means, cv_stds, plco_train_metrics, ukb_metrics, loss_curves

def nb_cross_validation_and_train(plco_data_path, ukb_data_path):
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']
    
    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_metrics = []
    loss_curves = []

    for train_index, val_index in kf.split(X_plco_train):
        X_train, X_val = X_plco_train.iloc[train_index], X_plco_train.iloc[val_index]
        y_train, y_val = y_plco_train.iloc[train_index], y_plco_train.iloc[val_index]
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred_val = model.predict_proba(X_val)[:, 1]
        cv_metrics.append(calculate_metrics(y_val, y_pred_val))

        # Calculate log loss for each iteration
        losses = []
        for i in range(1, 11):  # For Naive Bayes, just use 10 fixed iterations
            model.fit(X_train, y_train)
            y_pred_iter = model.predict_proba(X_val)[:, 1]
            losses.append(log_loss(y_val, y_pred_iter))
        loss_curves.append(losses)
    
    cv_metrics = np.array(cv_metrics)
    cv_means = np.mean(cv_metrics, axis=0)
    cv_stds = np.std(cv_metrics, axis=0)
    
    # Train on the entire PLCO dataset
    model.fit(X_plco_train, y_plco_train)
    
    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_plco_train)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']
    
    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)
    
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
    plt.savefig('graphs/ANN' + title + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_curves(male_history_list, female_history_list, title, model):
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(male_history_list):
        plt.plot(losses, label=f'Fold {i + 1}', alpha=0.3)
    for i, losses in enumerate(female_history_list):
        plt.plot(losses, label=f'Fold {i + 1}', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.5)
    plt.savefig(f'graphs/{model}/{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def table_of_metrics(title, model, cv_means=None, cv_stds=None, metrics_values=None):
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
    plt.savefig(f'metrics/{model}/{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(male_fpr_plco, male_tpr_plco, male_fpr_ukb, male_tpr_ukb, female_fpr_plco, female_tpr_plco, female_fpr_ukb, female_tpr_ukb, male_auc_plco, male_auc_ukb, female_auc_plco, female_auc_ukb, title, model):  
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})')
    plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})')
    plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})')
    plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(f'graphs/{model}/{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_evaluate_model(plco_data_path, ukb_data_path, model_type, title, filename):
    # Load and prepare PLCO data
    plco_data = pd.read_csv(plco_data_path)
    X_plco_train = plco_data.drop(columns=['lung'])
    y_plco_train = plco_data['lung']

    # Create and train the model
    if model_type == SVC:
        model = model_type(probability=True)
    else:
        model = model_type()
    model.fit(X_plco_train, y_plco_train)
    
    # Evaluate on PLCO training set
    y_pred_plco_train = model.predict_proba(X_plco_train)[:, 1]
    fpr_plco, tpr_plco, _ = roc_curve(y_plco_train, y_pred_plco_train)
    auc_plco = auc(fpr_plco, tpr_plco)

    # Calculate metrics for PLCO training data
    plco_train_metrics = calculate_metrics(y_plco_train, y_pred_plco_train)

    # Load and prepare UKB data
    ukb_data = pd.read_csv(ukb_data_path)
    X_ukb = ukb_data[X_plco_train.columns]
    y_ukb = ukb_data['lung']

    # Predict on UKB data
    y_pred_ukb = model.predict_proba(X_ukb)[:, 1]
    fpr_ukb, tpr_ukb, _ = roc_curve(y_ukb, y_pred_ukb)
    auc_ukb = auc(fpr_ukb, tpr_ukb)
    
    # Calculate metrics for UKB data
    ukb_metrics = calculate_metrics(y_ukb, y_pred_ukb)

    # I would like to plot the loss curve, but I don't know if every model type has a loss_curve_ attribute
    # plt.figure(figsize=(10, 10))
    # plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    # plt.plot(model.loss_curve_)
    # plt.xlabel('Epochs')
    # # plt.ylabel('Loss/Classification Score')
    # plt.title(title)
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()

    return (fpr_plco, tpr_plco, auc_plco), (fpr_ukb, tpr_ukb, auc_ukb), plco_train_metrics, ukb_metrics

# Plot all ROC curves on the same figure
def plot_auc(male_fpr_plco, male_tpr_plco, male_auc_plco,
             male_fpr_ukb, male_tpr_ukb, male_auc_ukb,
             female_fpr_plco, female_tpr_plco, female_auc_plco,
             female_fpr_ukb, female_tpr_ukb, female_auc_ukb,
             filename, title, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    plt.plot(male_fpr_plco, male_tpr_plco, label=f'PLCO Male (AUC = {male_auc_plco:.3f})', **kwargs)
    plt.plot(male_fpr_ukb, male_tpr_ukb, label=f'UKB Male (AUC = {male_auc_ukb:.3f})', **kwargs)
    plt.plot(female_fpr_plco, female_tpr_plco, label=f'PLCO Female (AUC = {female_auc_plco:.3f})', **kwargs)
    plt.plot(female_fpr_ukb, female_tpr_ukb, label=f'UKB Female (AUC = {female_auc_ukb:.3f})', **kwargs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Print metrics
def print_metrics(male_plco_train_metrics, male_ukb_metrics,
                  female_plco_train_metrics, female_ukb_metrics,
                  male_cv_means=None, male_cv_stds=None,
                  female_cv_means=None, female_cv_stds=None):
    if male_cv_means is not None:
        # Print cross-validation results for male data
        print("\nMale Cross-Validation Metrics (Mean ± Std):")
        print(f"Precision: {male_cv_means[0]:.4f} ± {male_cv_stds[0]:.4f}")
        print(f"F1 Score: {male_cv_means[1]:.4f} ± {male_cv_stds[1]:.4f}")
        print(f"Accuracy: {male_cv_means[2]:.4f} ± {male_cv_stds[2]:.4f}")
        print(f"PPV: {male_cv_means[3]:.4f} ± {male_cv_stds[3]:.4f}")
        print(f"NPV: {male_cv_means[4]:.4f} ± {male_cv_stds[4]:.4f}")
        print(f"MCC: {male_cv_means[5]:.4f} ± {male_cv_stds[5]:.4f}")
        print(f"Informedness: {male_cv_means[6]:.4f} ± {male_cv_stds[6]:.4f}")
        print(f"DOR: {male_cv_means[7]:.4f} ± {male_cv_stds[7]:.4f}")

    # Print training results for male data
    print("\nMale Training Metrics:")
    print("Precision: ", round(male_plco_train_metrics[0], 4))
    print("F1 Score: ", round(male_plco_train_metrics[1], 4))
    print("Accuracy: ", round(male_plco_train_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(male_plco_train_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(male_plco_train_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(male_plco_train_metrics[5], 4))
    print("Informedness: ", round(male_plco_train_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(male_plco_train_metrics[7], 4))

    # Print testing results for male data
    print("\nMale Testing Metrics on UKB Data:")
    print("Precision: ", round(male_ukb_metrics[0], 4))
    print("F1 Score: ", round(male_ukb_metrics[1], 4))
    print("Accuracy: ", round(male_ukb_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(male_ukb_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(male_ukb_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(male_ukb_metrics[5], 4))
    print("Informedness: ", round(male_ukb_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(male_ukb_metrics[7], 4))

    if female_cv_means is not None:
        # Print cross-validation results for female data
        print("\nFemale Cross-Validation Metrics (Mean ± Std):")
        print(f"Precision: {female_cv_means[0]:.4f} ± {female_cv_stds[0]:.4f}")
        print(f"F1 Score: {female_cv_means[1]:.4f} ± {female_cv_stds[1]:.4f}")
        print(f"Accuracy: {female_cv_means[2]:.4f} ± {female_cv_stds[2]:.4f}")
        print(f"PPV: {female_cv_means[3]:.4f} ± {female_cv_stds[3]:.4f}")
        print(f"NPV: {female_cv_means[4]:.4f} ± {female_cv_stds[4]:.4f}")
        print(f"MCC: {female_cv_means[5]:.4f} ± {female_cv_stds[5]:.4f}")
        print(f"Informedness: {female_cv_means[6]:.4f} ± {female_cv_stds[6]:.4f}")
        print(f"DOR: {female_cv_means[7]:.4f} ± {female_cv_stds[7]:.4f}")

    # Print training results for female data
    print("\nFemale Training Metrics:")
    print("Precision: ", round(female_plco_train_metrics[0], 4))
    print("F1 Score: ", round(female_plco_train_metrics[1], 4))
    print("Accuracy: ", round(female_plco_train_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(female_plco_train_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(female_plco_train_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(female_plco_train_metrics[5], 4))
    print("Informedness: ", round(female_plco_train_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(female_plco_train_metrics[7], 4))

    # Print testing results for female data
    print("\nFemale Testing Metrics on UKB Data:")
    print("Precision: ", round(female_ukb_metrics[0], 4))
    print("F1 Score: ", round(female_ukb_metrics[1], 4))
    print("Accuracy: ", round(female_ukb_metrics[2], 4))
    print("Positive Predictive Value (PPV): ", round(female_ukb_metrics[3], 4))
    print("Negative Predictive Value (NPV): ", round(female_ukb_metrics[4], 4))
    print("Matthews Correlation Coefficient (MCC): ", round(female_ukb_metrics[5], 4))
    print("Informedness: ", round(female_ukb_metrics[6], 4))
    print("Diagnostic Odds Ratio (DOR): ", round(female_ukb_metrics[7], 4))