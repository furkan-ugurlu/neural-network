# This script predicts the calorie content of Indian food based on ingredient quantities using a deep neural network.
# It demonstrates a full machine learning pipeline: data loading, preprocessing, model training, and evaluation.

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the processed Indian food nutrition dataset
file_path = "Indian_Food_Nutrition_Processed.csv"
data = pd.read_csv(file_path)
print("Dataset shape:", data.shape)
print("Missing values per column before imputation:\n", data.isnull().sum())

# Fill missing values: for columns with missing data, fill with half the mean value
for column in data.columns:
    if data[column].isnull().sum() > 0:
        mean_value = data[column].mean() / 2
        data[column].fillna(mean_value, inplace=True)
print("Missing values per column after imputation:\n", data.isnull().sum())

print("Available columns:", data.columns)

# Select features and target variable
features = ['Carbohydrates (g)', 'Protein (g)', 'Fats (g)', 'Free Sugar (g)']
X = data[features].copy()
y = data["Calories (kcal)"].copy()

# Standardize features and target for better neural network performance
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=42)
input_shape = (X_train.shape[1],)

# Build a deep neural network regression model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=input_shape),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(2, activation="relu"),
    layers.Dense(1, activation="linear"),  # Output layer for regression
])

# Compile the model with Adam optimizer and mean squared error loss
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# Train the model and store the training history
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=30,
    validation_split=0.16,
    verbose=1
)

# Predict on the test set
prediction = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Plot training and validation loss over epochs
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()