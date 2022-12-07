# Databricks notebook source
# MAGIC %md ## Model training workflow

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd

# COMMAND ----------

# MAGIC %md Create an MLflow experiment if one does not exist

# COMMAND ----------

def get_or_create_experiment(experiment_location: str) -> None:
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)


experiment_location = '/Shared/ml_production_experiment'
get_or_create_experiment(experiment_location)

mlflow.set_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md Train model and log to MLflow

# COMMAND ----------

# Convert to Pandas for scikit-learn training
data = spark.table('default.passenger_featurs_combined').toPandas()

# Split into training and test datasets
label = 'Survived'
features = [col for col in data.columns if col not in [label, 'PassengerId']]

X_train, X_test, y_train, y_test = train_test_split(data[features], data[label], test_size=0.25, random_state=123, shuffle=True)

# Categorize columns by data type
categorical_vars = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
numeric_vars = ['Age', 'FareRounded']
binary_vars = ['NameMultiple']

# Create the a pre-processing and modleing pipeline
binary_transform = make_pipeline(SimpleImputer(strategy = 'constant', fill_value = 'missing'))

numeric_transform = make_pipeline(SimpleImputer(strategy = 'most_frequent'))

categorical_transform = make_pipeline(SimpleImputer(missing_values = None, strategy = 'constant', fill_value = 'missing'), 
                                      OneHotEncoder(handle_unknown="ignore"))
  
transformer = ColumnTransformer([('categorial_vars', categorical_transform, categorical_vars),
                                 ('numeric_vars', numeric_transform, numeric_vars),
                                 ('binary_vars', binary_transform, binary_vars)],
                                  remainder = 'drop')

# Specify the model
# See Hyperopt for hyperparameter tuning: https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
model = xgb.XGBClassifier(n_estimators = 50, use_label_encoder=False)

classification_pipeline = Pipeline([("preprocess", transformer), ("classifier", model)])

# Fit the model, collect statistics, and log the model
with mlflow.start_run() as run:
  
  run_id = run.info.run_id
  #mlflow.xgboost.autolog()
    
  # Fit model
  classification_pipeline.fit(X_train, y_train)
  
  train_pred = classification_pipeline.predict(X_train)
  test_pred = classification_pipeline.predict(X_test)
  
  # Calculate validation statistics
  precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, train_pred, average='weighted')
  precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')
  
  decimals = 2
  validation_statistics = {"precision_training": round(precision_train, decimals),
                           "precision_testing": round(precision_test, decimals),
                           "recall_training": round(recall_train, decimals),
                           "recall_testing": round(recall_test, decimals),
                           "f1_training": round(f1_train, decimals),
                           "f1_testing": round(f1_test, decimals)}
  
  # Log the validation statistics
  mlflow.log_metrics(validation_statistics)
  
  # Fit final model
  final_model = classification_pipeline.fit(data[features], data[label])
  
  # Log the model and training data metadata
  mlflow.sklearn.log_model(final_model, artifact_path="model")
