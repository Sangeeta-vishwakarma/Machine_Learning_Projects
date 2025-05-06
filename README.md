# Machine_Learning_Project
## Laptop Price Predictor


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

Loaded data using pandas
Removed irrelevant columns and units from fields like Ram, Weight
Handled missing values and duplicates

## Exploratory Data Analysis (EDA)
Visualized the distribution of laptop brands, processor types, and RAM
Investigated correlations between features and the target variable (price)

## Feature Engineering
Extracted and transformed features such as:
PPI (Pixels Per Inch) from screen resolution
Simplified CPU and GPU brand
Converted categorical variables to numeric using Label Encoding/One-Hot Encoding

## Modeling
Trained and evaluated multiple regression models:
Linear Regression
Ridge and Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost
Stacking and Voting Regressors

## Evaluation Metrics

R² Score
RMSE (Root Mean Squared Error)
Comparison of model performances was conducted to finalize the best approach


## Why This Approach?
**Comprehensive Modeling**: By comparing several regression techniques, the project ensures a thorough evaluation of which model generalizes best on unseen data.

**Feature Engineering**: Domain-specific transformations like calculating PPI from resolution are used to boost model performance.

**Performance-centric**: Models are selected not only based on accuracy but also interpretability and speed, crucial in real-world applications.
