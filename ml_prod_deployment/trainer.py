# Databricks notebook source
# MAGIC %md ## Model training workflow
# MAGIC 
# MAGIC Note: Integration test should be a multi-task job that runs one notebook to train a model and another to load the model and perform inference. Move the "run_as_integration_test" to the inference notebook and add an assert statement.

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

from util import get_or_create_experiment, get_yaml_config

client = MlflowClient()

# COMMAND ----------

# MAGIC %md Set flag to determine if notebook is being run as part of an integration test

# COMMAND ----------

dbutils.widgets.dropdown("compare_model_registry_versions", "True", ["True", "False"])
compare_model_registry_versions = True if dbutils.widgets.get("compare_model_registry_versions") == "True" else False

# COMMAND ----------

# MAGIC %md Training parameters

# COMMAND ----------

training_config = get_yaml_config("training_scoring_config.yaml")

# COMMAND ----------

# MAGIC %md Model hyperparameters for prodution model

# COMMAND ----------

xgboost_params = {"n_estimators":25, 
                  "max_depth":5,
                  "use_label_encoder":False}

# COMMAND ----------

# MAGIC %md Create an MLflow Experiment and Model Registry entry if they do not already exist

# COMMAND ----------

get_or_create_experiment(training_config.mlflow_experiment_location)
mlflow.set_experiment(training_config.mlflow_experiment_location)

try:
  client.get_registered_model(training_config.mlflow_model_registry_name)
  print("A Model Registry entry with this name already exists")
except:
  client.create_registered_model(training_config.mlflow_model_registry_name)

# COMMAND ----------

# MAGIC %md Train model and log to MLflow

# COMMAND ----------

# Convert to Pandas for scikit-learn training
data = spark.table(training_config.feature_table_name).toPandas()

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
model = xgb.XGBClassifier(**xgboost_params)

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

# COMMAND ----------

# MAGIC %md Register new model as a version in the Model Registry

# COMMAND ----------

new_model_experiment = client.get_run(run_id)
new_model_validation_stat = new_model_experiment.data.metrics[training_config.validation_statistic_to_compare]
new_model_info = client.get_run(run_id).to_dictionary()
new_model_artifact_uri = new_model_info['info']['artifact_uri']

new_registered_model = client.create_model_version(
                         name = training_config.mlflow_model_registry_name,
                         source = new_model_artifact_uri + "/model",
                         run_id = run_id,
                         )

# COMMAND ----------

# MAGIC %md Transition new model to the Production stage in the Model Registry if appropriate criteria are met

# COMMAND ----------

if compare_model_registry_versions:
  """
  Confirm a Production model exists in the registry. Locate its Experiment run and capture its validation
  statistic. Compare the existing Production model's statistics to the new model. Promote the new model
  to the Production stage, archiving the existing model, only if the new model's validation statistic is
  at least as good as the current Production model.
  
  If an existing Production model does not exist, the new model will be transitioned to the Production stage.
  Similarly, if compare_model_registry_versions = False, the new model will be transitiond to the Production
  stage.
  """
  
  model_registry_versions = client.search_model_versions(f"name='{training_config.mlflow_model_registry_name}'")
  model_registry_stages = [run.current_stage for run in model_registry_versions]
  production_model_exists = True if 'Production' in model_registry_stages else False

  if production_model_exists:
    production_model_run_id = [run for run in model_registry_versions if run.current_stage == 'Production'][0].run_id
    production_model_experiment = client.get_run(production_model_run_id)
    production_model_validation_stat = production_model_experiment.data.metrics[training_config.validation_statistic_to_compare]

    print(f"""New model {training_config.validation_statistic_to_compare} score is {new_model_validation_stat} compare to production model's {production_model_validation_stat}""")

    if new_model_validation_stat >= production_model_validation_stat:
      print("Transitioning new model to Production and archiving existing Production model")
            
      promote_to_prod = client.transition_model_version_stage(name=training_config.mlflow_model_registry_name,
                                                        version = int(new_registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)
      
      update_new_model_version = client.update_model_version(name=training_config.mlflow_model_registry_name,
                                                             version=new_registered_model.version,
                                                             description = f"""Model's {training_config.validation_statistic_to_compare} score is {new_model_validation_stat} \
                                                                           compared to prior Production model's {production_model_validation_stat}; this model was transitioned \
                                                                           to Production""")

    else:
      print("New model will not be trasitioned to production")
      
      update_new_model_version = client.update_model_version(name=training_config.mlflow_model_registry_name,
                                                             version=new_registered_model.version,
                                                             description = f"""Model's {training_config.validation_statistic_to_compare} score is {new_model_validation_stat} \
                                                                           compared to prior Production model's {production_model_validation_stat}; this model was not transitioned \
                                                                           to Production""")
  else:
    print("A model in the Production stage does not exist in the Model Registry, promoting the new model to Production")
    
    promote_to_prod = client.transition_model_version_stage(name=training_config.mlflow_model_registry_name,
                                                          version = int(new_registered_model.version),
                                                          stage="Production",
                                                          archive_existing_versions=True)

    update_new_model_version = client.update_model_version(name=training_config.mlflow_model_registry_name,
                                                               version=new_registered_model.version,
                                                               description = f"""This model was not compared to a pre-existing Production model because one did not exist""")

else:
  print("Transition new model to Production without a comparison to existing Production model")
    
  promote_to_prod = client.transition_model_version_stage(name=training_config.mlflow_model_registry_name,
                                                          version = int(new_registered_model.version),
                                                          stage="Production",
                                                          archive_existing_versions=True)

  update_new_model_version = client.update_model_version(name=training_config.mlflow_model_registry_name,
                                                        version=new_registered_model.version,
                                                        description = f"""This model was not compared to a pre-existing Production because the comparison was not requested""")
