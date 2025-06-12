# Machine_Learning_Project
# 1. Laptop Price Predictor


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



# 2. Customer Churn Prediction Using Machine Learning
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


# 3. Breast Cancer Detection

This machine learning project aims to predict whether breast tumors are malignant (cancerous) or benign (non-cancerous) using the Breast Cancer Wisconsin (Diagnostic) dataset. The solution implements and evaluates multiple classification algorithms to identify the best-performing model for this critical medical prediction task.

## Dataset
- Features: 30 tumor characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension)
- Target Variable: diagnosis (M = Malignant, B = Benign)
- Samples: 569 (212 Malignant, 357 Benign)

## Key Features
- Comprehensive data preprocessing pipeline
- Exploratory Data Analysis (EDA) with visualizations
- Comparison of 8 machine learning models
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation with multiple metrics

## Data Preprocessing
- Removed null columns
- Encoded target variable (M=1, B=0)
- Scaled features using StandardScaler
- Train-test split (75%-25%)

## Modeling
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Classifier
5. K-Nearest Neighbors
6. Naive Bayes Classifier
7. AdaBoost Classifier
8. XGBoost Classifier

## Hyperparameter Tuning
Performed RandomizedSearchCV on:
- Random Forest Classifier
- XGBoost Classifier

## Results
### Model Performance Comparison
1. XGBoost (Tuned)
Accuracy: 97.9% | Sensitivity: 100% | Specificity: 94.6%

2. Random Forest
Accuracy: 97.2% | Sensitivity: 98.9% | Specificity: 94.5%

3. Support Vector Classifier
Accuracy: 95.8% | Sensitivity: 97.7% | Specificity: 92.7%

4. Logistic Regression
Accuracy: 95.1% | Sensitivity: 96.6% | Specificity: 92.6%

5. K-Nearest Neighbors
Accuracy: 95.1% | Sensitivity: 93.7% | Specificity: 97.9%

6. Naive Bayes
Accuracy: 94.4% | Sensitivity: 94.6% | Specificity: 94.1%

7. Decision Tree
Accuracy: 93.7% | Sensitivity: 97.6% | Specificity: 87.9%

8. AdaBoost
Accuracy: 93.0% | Sensitivity: 96.5% | Specificity: 87.7%

## Best Model Performance (XGBoost)
### Key Metrics:
- Accuracy: 97.9%
- Sensitivity: 100% (correctly identified all malignant cases)
- Specificity: 94.6%
- Precision (Malignant): 95%
- F1-Score (Malignant): 97%

## Conclusion
The tuned XGBoost classifier achieved the highest performance with:
- 97.9% accuracy
- 100% sensitivity (correctly identified all malignant cases)
- 94.6% specificity

This model demonstrates strong potential for assisting medical professionals in early detection of breast cancer. The high sensitivity is particularly valuable in medical diagnostics where false negatives (missing malignant tumors) have serious consequences.

# 4. Diabetes Classification

## Objective
This project aims to predict whether a person has diabetes (Class = 1) or not (Class = 0) using the PIMA Indian Diabetes Dataset. Several machine learning classification algorithms were implemented, evaluated, and compared based on performance metrics like accuracy, sensitivity, and specificity.

##  Dataset
- File Used: pima-diabetes-data.csv
- Shape: (767, 9)
- Features:
  - p_count: Patient count/index
  - PGC: Plasma Glucose Concentration
  - DBP: Diastolic Blood Pressure
  - Tricep: Skinfold thickness
  - SerumInsulin: 2-Hour serum insulin
  - BMI: Body Mass Index
  - DPF: Diabetes Pedigree Function
  - Age: Age in years
  - Class: Outcome (0 = Non-diabetic, 1 = Diabetic)

## Data Ingestion & Cleaning
- Read the dataset using pandas.
- Renamed the columns for readability.
- Checked for missing/null values.

## Data Exploration & Visualization
Used Seaborn and Matplotlib for:
- Count plot of the target class.
- Correlation heatmap.
- Pair plot (disabled hue due to misassignment).

## Data Preprocessing
- Separated features (X) and labels (y).
- Split data into train and test sets (80:20 ratio).
- Applied standard scaling using StandardScaler.

## Model Training
Trained the following models using the training set:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)

## Model Evaluation
Each model was evaluated using:
- Confusion Matrix
- Accuracy
- Sensitivity (Recall for class 1)
- Specificity (Recall for class 0)

## Model Performance (Training Accuracy)
- Logistic Regression:	          77.97%
- Decision Tree:	                100%
- Random Forest:	                99.84%
- Support Vector Classifier:	   78.14%
- K-Nearest Neighbors:	         84.66%
- Naive Bayes:	                 76.18%

#### Note: Decision Tree and Random Forest show signs of overfitting with near-perfect training accuracy

## Evaluation Metrics (on Test Data)
For each model, the following metrics were printed:
- Confusion Matrix
- Testing Accuracy
- Sensitivity (Recall for positive class)
- Specificity (Recall for negative class)

## Libraries Used
- pandas, numpy – Data manipulation
- matplotlib, seaborn – Visualization
- scikit-learn – Model building and evaluation
  - LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, GaussianNB, train_test_split, StandardScaler, confusion_matrix

## Conclusion
- KNN, Logistic Regression, and SVC gave a good trade-off between accuracy and overfitting.
- Random Forest and Decision Tree achieved high training accuracy but need regularization or pruning to avoid overfitting.
- Further evaluation using cross-validation and ROC-AUC can help in better model selection.


