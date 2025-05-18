# Indian Food Calorie Prediction with Deep Neural Network

This project demonstrates how to predict the calorie content of Indian food recipes based on their nutritional ingredients using a deep neural network built with TensorFlow/Keras.

## Overview

- **Goal:** Predict "Calories (kcal)" from nutritional features using a regression neural network.
- **Pipeline:** Data loading, missing value imputation, feature scaling, model training, evaluation, and visualization.

## Dataset

- **File:** `Indian_Food_Nutrition_Processed.csv`
- **Features used:**  
  - Carbohydrates (g)  
  - Protein (g)  
  - Fats (g)  
  - Free Sugar (g)  
- **Target:**  
  - Calories (kcal)
  
You can access dataset from here:
"https://www.kaggle.com/datasets/batthulavinay/indian-food-nutrition"

## How It Works

1. **Data Loading:** Reads the CSV file into a pandas DataFrame.
2. **Missing Value Handling:** Fills missing values in each column with half the mean of that column.
3. **Feature Selection:** Uses four main nutritional features as input.
4. **Scaling:** Standardizes both features and target for better neural network performance.
5. **Train/Test Split:** Splits the data into training and test sets.
6. **Model Building:** Constructs a deep neural network with several dense layers and ReLU activations.
7. **Training:** Trains the model using the Adam optimizer and mean squared error loss.
8. **Evaluation:** Prints Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test set.
9. **Visualization:** Plots training and validation loss over epochs.

## How to Run

1. Install required libraries:
    ```
    pip install pandas numpy scikit-learn matplotlib tensorflow
    ```
2. Make sure `Indian_Food_Nutrition_Processed.csv` is in the same folder as the script.
3. Run the script:
    ```
    python NeuralNetwork.py
    ```

## Example Output

- Prints dataset shape and missing value info.
- Shows MAE and MSE for test predictions.
- Displays a plot of training and validation loss.

## Model Architecture

The neural network structure is as follows:
```
Input (4 features) →
Dense(64, relu) →
Dense(64, relu) →
Dense(32, relu) →
Dense(16, relu) →
Dense(8, relu) →
Dense(4, relu) →
Dense(2, relu) →
Dense(1, linear)
```

## License

This project is for educational purposes.
