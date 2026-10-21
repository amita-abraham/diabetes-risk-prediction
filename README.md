# 🩺 Diabetes Risk Prediction Using Python

## 📌 Project Overview

This project develops a basic healthcare predictive analytics model to estimate whether a person is likely to have diabetes based on selected diagnostic measurements.

Using the Pima Indians Diabetes Dataset, the project applies **Machine Learning with Logistic Regression** to classify records into two categories: diabetes outcome 0 or outcome 1.

The project demonstrates the application of Python, data preprocessing, machine learning, and model evaluation to a healthcare-related problem.

## 🎯 Project Objectives

- Analyze historical healthcare data for diabetes prediction.
- Understand the relationship between patient measurements and diabetes outcomes.
- Prepare and preprocess data for machine learning.
- Develop a classification model using Logistic Regression.
- Evaluate model performance using multiple evaluation metrics.
- Visualize the results to make model performance easier to understand.

## 📊 Dataset Information

**Dataset:** Pima Indians Diabetes CSV

**Source:** [Kaggle — Pima Indians Diabetes CSV](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)

The commonly used version of this dataset contains 768 records, eight predictor variables, and one target variable. The dataset represents women aged 21 and above of Pima Indian heritage.

### Features and Target Variable

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function, reflecting family-history-related information |
| Age | Age in years |
| Outcome | Target variable: 0 or 1 |

**Target variable interpretation:**
- `0` — Diabetes-negative outcome in the dataset
- `1` — Diabetes-positive outcome in the dataset

*Note: These labels represent the dataset's recorded outcomes and are not a substitute for a clinical diagnosis.*

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Joblib** — Saving the trained model
- **Visual Studio Code** — Development environment
- **Git and GitHub** — Version control and project hosting

## ⚙️ Project Workflow

### 1. Data Loading
Load the diabetes dataset from a CSV file using Pandas and assign the appropriate column names if the file does not contain a header row.

### 2. Data Inspection
Inspect the dataset's structure, dimensions, data types, and target variable to understand the available information.

### 3. Data Preprocessing
Prepare the features and target variable for model training. The workflow uses median imputation to handle missing values and standard scaling to help put numerical features on comparable scales.

### 4. Train-Test Split
Divide the dataset into training and testing subsets using an 80:20 split. Stratification helps preserve the target-class proportions in both subsets.

### 5. Model Training
Train a **Logistic Regression** classification model using the training data.

### 6. Model Evaluation
Evaluate the trained model using the held-out test data and multiple performance metrics.

### 7. Visualization and Outputs
Generate evaluation reports and charts to help interpret the model's performance and the relationship between input features and predictions.

### 8. Model Saving
Save the trained preprocessing and prediction pipeline using Joblib for potential reuse.

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification. In this project, it estimates the probability of each record belonging to one of the two diabetes outcome classes.

The model uses the following input features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The `Outcome` column is used as the target variable during training.

## 📈 Model Evaluation Metrics

The model is evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of all test predictions that are correct |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases identified by the model |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures how well the model separates the two classes across classification
