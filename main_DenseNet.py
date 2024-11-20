import pandas as pd
import numpy as np
import tensorflow as tf
import dask.dataframe as dd
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Load CSV files using Dask
train_data_full = dd.read_csv('train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv').iloc[:, 1:]  # Skip the index column
train_data = train_data_full.sample(frac=0.01).compute()  # Sample 1% of the data training data

test_set_1 = dd.read_csv('val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv').iloc[:, 1:].compute()  # Skip the index column
test_set_2 = dd.read_csv('v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv').iloc[:, 1:].compute()  # Skip the index column

# Extract features and labels
X_train = train_data.iloc[:, 3:].values  # As features start from column 4
y_train = train_data['label'].values

X_test1 = test_set_1.iloc[:, 3:].values
y_test1 = test_set_1['label'].values

X_test2 = test_set_2.iloc[:, 3:].values
y_test2 = test_set_2['label'].values

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the number of classes
num_classes = len(np.unique(y_train))

# Build DenseNet model
def build_densenet(input_shape=(1023,)):  # Adjusting the input shape to match your data
    inputs = tf.keras.Input(shape=input_shape)  # Define input layer
    x = Dense(1023, activation='relu')(inputs)  # First Dense layer
    x = Dropout(0.65)(x)  # Dropout layer for regularization
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer
    return Model(inputs=inputs, outputs=predictions)


# Build models
densenet_model = build_densenet()

# Compile models
densenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train models
print("Training DenseNet model...")
history = densenet_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


# Calculate and print validation accuracy on y_train
y_val_pred = np.argmax(densenet_model.predict(X_val), axis=1)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy on y_train: {val_accuracy:.4f}")

# Generate classification report for validation set
print("\nClassification Report for Validation Set:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test sets
densenet_test_loss_1, densenet_test_acc_1 = densenet_model.evaluate(X_test1, y_test1)
densenet_test_loss_2, densenet_test_acc_2 = densenet_model.evaluate(X_test2, y_test2)

print("DenseNet Test Set 1 (val_eva02) Accuracy:", densenet_test_acc_1)
print("DenseNet Test Set 2 (v2_eva02) Accuracy:", densenet_test_acc_2)


# Predictions on test sets
y_pred_test1 = np.argmax(densenet_model.predict(X_test1), axis=1)
y_pred_test2 = np.argmax(densenet_model.predict(X_test2), axis=1)


#Generate classification reports with following codes
print("\nClassification Report for Test Set 1:")
print(classification_report(y_test1, y_pred_test1))

print("\nClassification Report for Test Set 2:")
print(classification_report(y_test2, y_pred_test2))

# # Set the sample size to 10
# sample_size = 10  

# # Sample indices for Test Set 1
# indices_test1 = random.sample(range(len(y_test1)), sample_size)

# # Sampled true and predicted labels for Test Set 1
# y_test1_sample = y_test1[indices_test1]
# y_pred_test1_sample = y_pred_test1[indices_test1]

# # Sample indices for Test Set 2
# indices_test2 = random.sample(range(len(y_test2)), sample_size)


# # Sampled true and predicted labels for Test Set 2
# y_test2_sample = y_test2[indices_test2]
# y_pred_test2_sample = y_pred_test2[indices_test2]

# # Generate confusion matrices
# cm_test1 = confusion_matrix(y_test1_sample, y_pred_test1_sample)
# cm_test2 = confusion_matrix(y_test2_sample, y_pred_test2_sample)

# # Plotting function for confusion matrices
# def plot_confusion_matrix(cm, title):
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(title)
#     plt.show()

# # Plot confusion matrices for sampled data
# plot_confusion_matrix(cm_test1, 'Confusion Matrix for Sampled Test Set 1')
# plot_confusion_matrix(cm_test2, 'Confusion Matrix for Sampled Test Set 2')

# Analyze performance gap by following codes
# Plotting the distribution of one feature (example: feature 4) in both test sets
# feature_index = 859  # Adjust based on which feature you want to analyze
# plt.figure(figsize=(12, 6))
# sns.histplot(data=test_set_1, x=test_set_1.columns[feature_index], kde=True, color='blue', label='Test Set 1', stat='density', alpha=0.5)
# sns.histplot(data=test_set_2, x=test_set_2.columns[feature_index], kde=True, color='orange', label='Test Set 2', stat='density', alpha=0.5)
# plt.title(f'Distribution of Feature {feature_index} in Test Sets')
# plt.xlabel('Feature Value')
# plt.ylabel('Density')
# plt.legend()
# plt.show()



