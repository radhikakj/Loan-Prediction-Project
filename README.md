# Loan Prediction Project

## Overview
This project aims to predict the likelihood of loan approval based on various applicant features such as income, education, credit history, etc. Machine learning algorithms are employed to build predictive models using a loan dataset.

## Dataset Description
The dataset contains information about loan applicants, including demographic details, financial information, and loan approval status. It consists of X samples and Y features.

## Analysis Steps
1. Data Preprocessing: Handling missing values, encoding categorical features, and feature scaling.
2. Exploratory Data Analysis: Visualizing data distributions, correlations, and exploring relationships between features.
3. Model Building: Implementing machine learning algorithms such as Linear Regression, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision Trees, and Random Forests.
4. Model Evaluation: Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.

## Results
- Linear Regression: MSE = 0.1636, R-squared = 0.2803
- Logistic Regression: Accuracy = 0.7886, Precision = 0.7596, Recall = 0.9875, F1-score = 0.8587
- K-Nearest Neighbors: Accuracy = 0.5772, Precision = 0.6321, Recall = 0.8375, F1-score = 0.7204
- Support Vector Machines: Accuracy = 0.7724
- Decision Trees: Accuracy = 0.7236
- Random Forests (before tuning): Accuracy = 0.7642
- Random Forests (after tuning): Accuracy = 0.7480

## Usage Instructions
1. Clone the repository.
2. Install the required dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn).
3. Run the Jupyter notebook to execute the code.
4. Follow the instructions within the notebook to reproduce the analysis.

## Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
