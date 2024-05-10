import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

# Load data
data = pd.read_csv('Male_Lung_Data_Greg_imputed.csv')

# Separate features and target
X = data.drop(columns=['lung'])
y = data['lung']

# Split data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Model
model = keras.models.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=[X_train.shape[1]],
                       kernel_initializer=keras.initializers.glorot_normal(),
                       bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(80, activation='relu',
                       kernel_initializer=keras.initializers.glorot_normal(),
                       bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# Loss and optimizer
loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1)
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.AUC(from_logits=False),
           tf.keras.metrics.Precision(name='PPV'),
           tf.keras.metrics.Recall(name='Sensitivity')]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# Training
batch_size = 1024
epochs = 10

model.fit(X_train.to_numpy(), y_train.to_numpy(), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# Evaluation
model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)

# Plot ROC curve
y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='lung (AUC = {:.3f})'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate (Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('PLCO Male: ROC curve (Neural Network)')
plt.legend(loc='lower right')
plt.savefig('male_lung_ROC.png', dpi=300, bbox_inches='tight')
plt.show()

# Load UKB data
cancers = ['lung']
data = pd.read_csv('Data Files/imputedMaleDataGreg2.csv')

# Assuming 'lung' column contains the target variable
# Convert the target variable to binary
#data['lung'] = (data['lung'] == 1).astype(int)

x = data[X.columns]  # Assuming 'X' contains the same features as 'x_train'

# Predict using the trained model
y_pred = model.predict(x)

plt.figure(1, figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
for i in np.arange(len(cancers)):
    fpr, tpr, thresholds = roc_curve(data['lung'], y_pred[:, i])  # Ensure 'lung' is binary
    plt.plot(fpr, tpr, label=cancers[i] + ('(auc = {:.3f})'.format(auc(fpr, tpr))))
plt.xlabel('False positive rate (Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('Male Lung Cancer UK Biobank Testing: ROC curve')
plt.legend(loc='lower right')
plt.savefig('maleCancerROCUKBiobank.png', dpi=300, bbox_inches='tight')
plt.show()