# Machine_Learning_Project
## 1. Laptop Price Predictor


## Objective
To build a machine learning model that accurately predicts the price of a laptop based on its specifications such as brand, processor, RAM, GPU, and other hardware attributes.


## Summary
This project demonstrates an end-to-end machine learning pipeline involving:
* Data collection and preprocessing
* Exploratory data analysis (EDA)
* Feature engineering
* Model training and evaluation using various regression techniques
* Performance comparison and selection of the best model

The project helps understand how different laptop features contribute to price determination, using both traditional and ensemble machine learning models.


# Project Description


The dataset used was sourced from Kaggle and contains specifications of various laptops along with their respective prices.
## Data Loading & Cleaning
* Loaded data using pandas
* Removed irrelevant columns and units from fields like Ram, Weight
* Handled missing values and duplicates

## Exploratory Data Analysis (EDA)
* Visualized the distribution of laptop brands, processor types, and RAM
* Investigated correlations between features and the target variable (price)

## Feature Engineering
Extracted and transformed features such as:
* PPI (Pixels Per Inch) from screen resolution
* Simplified CPU and GPU brand
* Converted categorical variables to numeric using Label Encoding/One-Hot Encoding

## Modeling
Trained and evaluated multiple regression models:
* Linear Regression
* Ridge and Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost
* Stacking and Voting Regressors

## Evaluation Metrics
* R² Score
* RMSE (Root Mean Squared Error)
* Comparison of model performances was conducted to finalize the best approach


## Why This Approach?
**Comprehensive Modeling**: By comparing several regression techniques, the project ensures a thorough evaluation of which model generalizes best on unseen data.

**Feature Engineering**: Domain-specific transformations like calculating PPI from resolution are used to boost model performance.

**Performance-centric**: Models are selected not only based on accuracy but also interpretability and speed, crucial in real-world applications.


## Results
* The best-performing model achieved an R² score of ~89%, indicating high prediction accuracy.
* Feature importance analysis revealed that screen resolution (PPI), RAM size, and CPU brand were top contributors to price prediction.



## 2. Customer Churn Prediction Using Machine Learning
This project builds a binary classification model to predict whether a customer will churn (i.e., leave a bank) using a neural network built with TensorFlow/Keras. It performs data preprocessing, trains a deep learning model, and evaluates its performance using accuracy metrics and visualizations.

## Dataset
The dataset used is Churn_Modelling.csv, containing records of 10,000 customers and 14 attributes including:
- Customer details: Geography, Gender, Age, Balance, Tenure, etc.
- Target variable: Exited — 1 if the customer left the bank, 0 otherwise.

## Data Loading and Exploration
- Loaded dataset using pandas.
- Inspected the shape, info, and verified there are no duplicate records.
- Analyzed class distribution for Exited, Gender, and Geography.

## Data Preprocessing
- Dropped irrelevant columns: RowNumber, CustomerId, Surname.
- Applied One-Hot Encoding to Geography and Gender.
- Split data into features X and target y.
- Performed train-test split (80% train, 20% test).
- Scaled features using StandardScaler.

## Model Building
- Used Sequential model from Keras.
- Two hidden layers allow the model to approximate more complex functions.
- ReLU activation helps avoid vanishing gradient problem.
- Sigmoid in output gives probabilities for binary classification.
- Architecture:
  - Input Layer: 11 nodes
  - Hidden Layer 1: 11 nodes, ReLU
  - Hidden Layer 2: 11 nodes, ReLU
  - Output Layer: 1 node, Sigmoid (for binary classification)

## Model Compilation & Training
Compiled with:
- Loss: binary_crossentropy
  - Ideal for binary classification tasks like churn prediction.
  - Penalizes incorrect predictions more when the model is confident but wrong, encouraging better calibration.
- Optimizer: Adam
  - Adaptive learning rate optimizer—combines the best of RMSProp and SGD.
  - Works well for most problems, particularly when training deep networks.
- Metrics: accuracy
  - Accuracy is a simple and intuitive metric for balanced binary classification problems.
  - Plotting loss and accuracy curves helps:
    - Monitor training process.
    - Detect overfitting or underfitting.
Trained on the scaled training data.

## Evaluation
- Predicted churn probability on test data.
- Converted probabilities to class labels using a threshold of 0.5.
- Evaluated model performance using accuracy_score.
- Plotted training history:
  - Loss vs Epoch
  - Accuracy vs Epoch

