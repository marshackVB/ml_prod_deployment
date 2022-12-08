import mlflow
import yaml
from argparse import Namespace

def get_or_create_experiment(experiment_location: str) -> None:
  """
  Given a DBFS directory, create an MLflow Experiment using that location if one
  does not already exist
  """
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)
  
  
def get_yaml_config(config_file:str) -> Namespace:
  """
  Loads paramerts from a yaml configuration file in a
  Namespace object.

  Args:
    config_file: A path to a yaml configuration file.

  Returns:
    A Namespace object that contans configurations referenced
    in the program.
  """
  stream = open(config_file, 'r')
  config_dict = yaml.load(stream, yaml.SafeLoader)
  
  for parameter, value in config_dict.items():
    print("{0:30} {1}".format(parameter, value))
    
  config = Namespace(**config_dict)
  
  return config