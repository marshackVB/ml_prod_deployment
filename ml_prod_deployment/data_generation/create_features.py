# Databricks notebook source
# MAGIC %md #### Create the feature tables from raw data

# COMMAND ----------

from pyspark.sql.functions import col
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType
import pandas as pd
from feature_transformations import compute_passenger_ticket_features, compute_passenger_demographic_features

# COMMAND ----------

# MAGIC %md Create Pyspark schemas

# COMMAND ----------

passenger_ticket_types = [('PassengerId',     StringType()),
                          ('Ticket',          StringType()),
                          ('Fare',            DoubleType()),
                          ('Cabin',           StringType()),
                          ('Embarked',        StringType()),
                          ('Pclass',          StringType()),
                          ('Parch',           StringType())]

passenger_demographic_types = [('PassengerId',StringType()),
                               ('Name',       StringType()),
                               ('Sex',        StringType()),
                               ('Age',        DoubleType()),
                               ('SibSp',      StringType())]

passenger_label_types = [('PassengerId',StringType()),
                         ('Survived',   IntegerType())]
  
  
def create_schema(col_types):
  struct = StructType()
  for col_name, type in col_types:
    struct.add(col_name, type)
  return struct

passenger_ticket_schema =      create_schema(passenger_ticket_types)
passenger_dempgraphic_schema = create_schema(passenger_demographic_types)
passenger_label_schema =       create_schema(passenger_label_types)

# COMMAND ----------

# MAGIC %md Create Pyspark DataFrames

# COMMAND ----------

def create_pd_dataframe(csv_file_path, schema):
    df = pd.read_csv(csv_file_path)
    return spark.createDataFrame(df, schema = schema)
  
  
passenger_ticket_raw = create_pd_dataframe('../../data/passenger_ticket.csv', passenger_ticket_schema)
passenger_demographic_raw = create_pd_dataframe('../../data/passenger_demographic.csv', passenger_dempgraphic_schema)
passenger_labels = create_pd_dataframe('../../data/passenger_labels.csv', passenger_label_schema)

# COMMAND ----------

# MAGIC %md Create passenger features

# COMMAND ----------

passenger_ticket_features = compute_passenger_ticket_features(passenger_ticket_raw)
passenger_demographic_features = compute_passenger_demographic_features(passenger_demographic_raw)

primary_key = 'PassengerId'
combined_features = (passenger_ticket_features.join(passenger_demographic_features, [primary_key], 'left')
                                              .join(passenger_labels, [primary_key], 'left'))

display(combined_features)

# COMMAND ----------

# MAGIC %md Persist feature table to Delta

# COMMAND ----------

combined_features.write.mode('overwrite').format('delta').saveAsTable('default.passenger_featurs_combined')
