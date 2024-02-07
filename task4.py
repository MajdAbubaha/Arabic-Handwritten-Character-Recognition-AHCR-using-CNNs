# import numpy as np
# from keras import layers, models
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# from keras.callbacks import ReduceLROnPlateau
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from keras.models import load_model
# import tensorflow as tf
# from keras.optimizers import Adam
# import warnings
#
# # Filter warnings
# warnings.filterwarnings('ignore')
#
# # Read train data and labels
# train = pd.read_csv("./Arabic Handwritten Characters Dataset CSV/csvTrainImages 13440x1024.csv")
# train_label = pd.read_csv("./Arabic Handwritten Characters Dataset CSV/csvTrainLabel 13440x1.csv",
#                           header=None,
#                           names=['label'])
#
# # Read test data and labels
# test = pd.read_csv("./Arabic Handwritten Characters Dataset CSV/csvTestImages 3360x1024.csv")
# test_label = pd.read_csv("./Arabic Handwritten Characters Dataset CSV/csvTestLabel 3360x1.csv",
#                          header=None,
#                          names=['label'])
#
# # Normalize and reshape train data
# X_train = train.values.reshape(-1, 32, 32, 1) / 255.0
# X_train = tf.image.resize(X_train, [28, 28])
#
# # Adjust labels
# train_label['label'] = train_label['label'] - 1
#
# # Encode labels
# num_classes = len(train_label['label'].unique())
# Y_train = to_categorical(train_label['label'], num_classes=num_classes)[:-1]
#
# # Convert labels to numpy arrays
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
#
# # Split the train and validation sets
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2)
#
#
# # Load the pre-trained model
# file = './mnist.h5'
# pretrained_model = load_model(file)
#
# # Modify the model
# if isinstance(pretrained_model, models.Sequential):
#     pretrained_model.pop()
#     pretrained_model.add(
#         layers.Dense(28, activation='softmax', name='new_output_layer'))
#
# # Define optimizer and compile model
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# pretrained_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#
# # Define ReduceLROnPlateau callback
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
#
# # Fit the model
# history = pretrained_model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val),
#                                callbacks=[reduce_lr])
#
# # Plot loss and accuracy curves
# plt.figure(figsize=(12, 12))
# for i, metric in enumerate(['loss', 'val_loss', 'accuracy', 'val_accuracy']):
#     plt.subplot(2, 2, i+1)
#     plt.plot(history.history[metric], label=metric.capitalize())
#     plt.title(metric.capitalize())
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss' if 'loss' in metric else 'Accuracy')
#     plt.legend()
# plt.show()
#
# # # Evaluate model on validation data
# # Y_pred = pretrained_model.predict(X_val)
# # Y_pred_classes = np.argmax(Y_pred, axis=1)
# # Y_true = np.argmax(Y_val, axis=1)
# # confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# #
# # # Plot confusion matrix
# # plt.figure(figsize=(15, 15))
# # sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.title("Confusion Matrix")
# # plt.show()
#
# # Calculate accuracy, precision, and recall
# # accuracy = accuracy_score(Y_true, Y_pred_classes)
# # precision = precision_score(Y_true, Y_pred_classes, average='weighted')
# # recall = recall_score(Y_true, Y_pred_classes, average='weighted')
# # print("Accuracy:", accuracy)
# # print("Precision:", precision)
# # print("Recall:", recall)
#
# # Normalize and reshape test data
# X_test = test.values.reshape(-1, 32, 32, 1) / 255.0
# X_test = tf.image.resize(X_test, [28, 28])
#
# # Adjust labels
# test_label['label'] = test_label['label'] - 1
#
# # Encode true labels
# Y_test = to_categorical(test_label['label'], num_classes=num_classes)[:-1]
#
# # Predictions on testing data
# Y_pred_test = pretrained_model.predict(X_test)
# Y_pred_classes_test = np.argmax(Y_pred_test, axis=1)
# Y_true_test = np.argmax(Y_test, axis=1)
# confusion_mtx_test = confusion_matrix(Y_true_test, Y_pred_classes_test)
#
# # Plot test confusion matrix
# plt.figure(figsize=(15, 15))
# sns.heatmap(confusion_mtx_test, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Test Confusion Matrix")
# plt.show()
#
# # Calculate accuracy, precision, and recall for testing data
# accuracy_test = accuracy_score(Y_true_test, Y_pred_classes_test)
# precision_test = precision_score(Y_true_test, Y_pred_classes_test, average='weighted')
# recall_test = recall_score(Y_true_test, Y_pred_classes_test, average='weighted')
# print("Accuracy on Testing Data:", accuracy_test)
# print("Precision on Testing Data:", precision_test)
# print("Recall on Testing Data:", recall_test)

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
from keras.models import load_model
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
Y_train = train_label.values.squeeze()

X_train = tf.image.resize(X_train, [28, 28])
X_train = X_train.numpy()

# Convert labels to one-hot encoding
number_of_classes = len(np.unique(train_label))
min_label = np.min(Y_train)
Y_train = tf.keras.utils.to_categorical(Y_train - min_label, number_of_classes)


# Split the data into training and validation sets
X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Load the pre-trained model
model_path = './mnist.h5'
model = load_model(model_path)

model.summary()

# Freeze convolutional layers
for layer in model.layers:
    if isinstance(layer, layers.Conv2D):
        layer.trainable = False

# Remove the last layer
model.pop()
# Add new output layer
model.add(layers.Dense(28, activation='softmax', name='new_output_layer'))

# Display model summary after modification
model.summary()

# if isinstance(model, models.Sequential):
#     model.pop()  # Remove the last layer
#     model.add(layers.Dense(28, activation='softmax', name='new_output_layer'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=42, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr])

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
y_test = test_label.values.squeeze()

X_test = tf.image.resize(X_test, [28, 28])
X_test = X_test.numpy()


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
