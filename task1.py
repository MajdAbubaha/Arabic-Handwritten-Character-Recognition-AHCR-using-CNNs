from keras.callbacks import ReduceLROnPlateau
import seaborn as sns
from keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import random


# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Filter warnings
warnings.filterwarnings('ignore')

# Load the datasets
train = pd.read_csv("./Arabic Handwritten Characters Dataset CSV/csvTrainImages 13440x1024.csv",
                    header=None)
train_label = pd.read_csv('./Arabic Handwritten Characters Dataset CSV/csvTrainLabel 13440x1.csv',
                          header=None)
test = pd.read_csv('./Arabic Handwritten Characters Dataset CSV/csvTestImages 3360x1024.csv',
                   header=None)
test_label = pd.read_csv('./Arabic Handwritten Characters Dataset CSV/csvTestLabel 3360x1.csv',
                         header=None)

# Preprocess training data
X_train = train.values.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
Y_train = train_label.values

# Convert labels to one-hot encoding
number_of_classes = len(np.unique(train_label))
min_label = np.min(Y_train)
Y_train = tf.keras.utils.to_categorical(Y_train - min_label, number_of_classes)


# Split the data into training and validation sets
X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(number_of_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(X_train, Y_train, epochs=20, batch_size=42, validation_data=(X_val, y_val), callbacks=[reduce_lr])

# Plot the training history
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Preprocess testing data
X_test = test.values.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
y_test = test_label.values

y_test = tf.keras.utils.to_categorical(y_test - min_label, number_of_classes)

# Evaluate the model on the test set
y_pred_test = model.predict(X_test)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_true_test = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy_test = accuracy_score(y_true_test, y_pred_classes_test)
precision_test = precision_score(y_true_test, y_pred_classes_test, average='weighted')
recall_test = recall_score(y_true_test, y_pred_classes_test, average='weighted')
f1_score_test = f1_score(y_true_test, y_pred_classes_test, average='weighted')

# Print metrics
print("Accuracy on Testing Data:", accuracy_test)
print("Precision on Testing Data:", precision_test)
print("Recall on Testing Data:", recall_test)
print("F1 Score on Testing Data:", f1_score_test)

# Plot confusion matrix
confusion_mtx_test = confusion_matrix(y_true_test, y_pred_classes_test)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_test, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Test Confusion Matrix')
plt.show()
