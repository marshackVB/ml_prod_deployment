# Databricks notebook source
# MAGIC %md #### Model scoring workflow

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql.functions import struct, col
from util import get_or_create_experiment, get_yaml_config

client = MlflowClient()

# COMMAND ----------

dbutils.widgets.dropdown("run_as_integration_test", "False", ["True", "False"])
run_as_integration_test = True if dbutils.widgets.get("run_as_integration_test") == "True" else False
print(run_as_integration_test)

# COMMAND ----------

scoring_config = get_yaml_config("training_scoring_config.yaml")

# COMMAND ----------

# MAGIC %md Load production model

# COMMAND ----------

def get_run_id(model_name, stage='Production'):
  """Get production model id from Model Registry"""
  
  prod_run = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  
  return prod_run.run_id

production_run_id = get_run_id(scoring_config.mlflow_model_registry_name, stage='Production')
production_model_uri = f"runs:/{production_run_id}/model"

print(f"Loading Production model from Registry, {scoring_config.mlflow_model_registry_name}, with run_id: {production_run_id}")

loaded_model = mlflow.pyfunc.load_model(f"runs:/{production_run_id}/model")

# COMMAND ----------

# MAGIC %md Score records and write to Delta

# COMMAND ----------

features_df = (spark.table(scoring_config.feature_table_name))
non_feature_columns = ["Survived", "PassengerId"]
features_columns = [col for col in features_df.columns if col not in non_feature_columns]

predictions_df = features_df.withColumn('predictions', loaded_model(struct(*map(col, features_columns))))

display(predictions)

# Change to 'append'
predictions_df.write.mode('overwrite').format('delta').saveAsTable(scoring_config.predictions_table)

# COMMAND ----------

# MAGIC %md Execute tests

# COMMAND ----------

def test_counts():
  # Confirm precitions table contains predictions
  _count = predictions_df.count()
  assert _count > 0

# COMMAND ----------

test_counts()
