# IntelliEstate
IntelliEstate: Real Estate Price Predictor 🏡 
This project, IntelliEstate, is an end-to-end machine learning pipeline that predicts real estate prices. The goal was to build a robust and accurate model to provide real-time property valuations. The entire pipeline is built using Python.

# Dataset's Features
CRIM: Per capita crime rate by town.

ZN: Proportion of residential land zoned for lots over 25,000 sq.ft..

INDUS: Proportion of non-retail business acres per town.

NOX: Nitric oxides concentration (parts per 10 million).

RM: Average number of rooms per dwelling.

AGE: Proportion of owner-occupied units built prior to 1940.

DIS: Weighted distances to five Boston employment centers.

RAD: Index of accessibility to radial highways.

TAX: Full-value property-tax rate per $10,000.

PTRATIO: Pupil-teacher ratio by town.

LSTAT: Percentage of the population with lower status.

MEDV: The target variable, representing the median value of owner-occupied homes in $1000s.

# Project Features ✨
End-to-End ML Pipeline: A complete system covering data ingestion, preprocessing, model training, evaluation, and deployment.

Robust Predictive Model: The model uses historical data to predict house prices with high accuracy, validated through cross-validation.

Scalable Solution: The pipeline is built to handle new data and provide valuations on demand.

Data Visualization: Exploratory Data Analysis (EDA) was performed using matplotlib and seaborn to understand data distributions and relationships between features.

# Technical Stack 🛠️
Python: The core programming language for the entire project.

scikit-learn: Used for building and evaluating machine learning models.

pandas & numpy: Essential libraries for data manipulation, analysis, and numerical operations.

joblib: Used to save the trained machine learning model for later use.

Matplotlib and Seaborn: For exploratory data analysis.

# Methodology 📊
Data Preprocessing: The pipeline began by cleaning and preparing the raw data. This included handling missing values using imputation and creating a new feature, TAXRM (a ratio of property tax to the number of rooms), to improve predictive power.

Model Training: Several machine learning models were trained and compared, including:

Linear Regression

Decision Tree

Random Forest Regressor

Model Evaluation: To ensure the model's performance was reliable, 10-fold cross-validation was used. The Random Forest Regressor was selected as the optimal model, achieving a Mean Root Mean Squared Error (RMSE) of 3.49, which indicates a high degree of predictive accuracy.

