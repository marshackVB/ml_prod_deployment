## Deploying and testing model training and inference workflows across Databricks Workspaces 
This repository contains an example machine learning project implementation based on Databricks Repos and [dbx](https://dbx.readthedocs.io/en/latest/). The dbx library's cli deployment functionality is used to deploy training and inference Jobs across Workspaces. In addition, this project contains an integration test to test the end to end training and inference workflows. This test can be run in a development Workspace before deploying training and inference jobs to a production Workspace.

The dbx library makes it easy to deploy jobs and test code using its cli tools. The .dbx folder contains information associated with Databricks CLI profiles that can be used to deploy jobs across Workspaces.

The conf/deployment.yml file within dbx projects provides a way to specify various Jobs configurations, including [multi-task jobs](https://learn.microsoft.com/en-us/azure/databricks/workflows/jobs/jobs).

See the [dbx documentation and tutorials](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#preparing-the-local-environment). 

### Prerequisites to running this project 
 - Install the [Databricks CLI](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/).
 - Configure the [Databricks CLI profiles](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/#--connection-profiles) for each workspace in which you want to deploy this project.
 - Execute the [dbx installation instructions](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#preparing-the-local-environment).
 - Activate the conda environment configured during the previous dbx setup step.

 ### Running the project 
  - Create an empty git repository in any git server supported by Databricks.
  - Clone this project to your local machine and push a copy to your own remote git repository using the below method.
   ```
   git clone https://github.com/marshackVB/ml_prod_deployment
   cd ml_prod_deployment/
   rm -rf .git
   git init
   git remote add origin https://github.com/<your_user_or_org_name>/<your_repo_name>.git
   git add .
   git commit -m 'initial commit'
   git push origin master
   ```
  - Adjust the dbx project configuration file at .dbx/project.json to reflect your own Databricks CLI profile names.
  - Push these updates to your remote git repository.
  - Clone the repository into a Databricks Repo. 
  - Run the commands below from your local terminal for remote execution in one or more Databricks Workspaces.


  ### Creating the datasets
  In your terminal, cd to this project's directory and execute the below commands. The job will be executed in the default Databricks environment, which is specified in the .dbx/project.json file. To run in another environment, add the --environment parameter and specify another environment name defined in the project.json file.
  ```
  dbx deploy ml-prod-deployment-create-features --assets-only
  dbx launch ml-prod-deployment-create-features --from-assets --trace
  ```

  ### Training the model

  ```
  dbx deploy ml-prod-deployment-train-model --assets-only
  dbx launch ml-prod-deployment-train-model --from-assets --trace
  ```

  ### Creating a recurring inference job
  ```
  dbx deploy ml-prod-deployment-score-records
  dbx launch ml-prod-deployment-score-records
  ```
  The above deploy command creates a persistent job in the Databricks Jobs UI that runs on a schedule defined in the conf/deployment.yml file. Calling 'dbx launch' then starts a run of this Job, outside of its normal run schedule.


  ### Deploying jobs across Workspaces
  Deploying across Workspaces is as easy as setting the --environment parameters when calling dbx commands, such as those above.
  ```
  dbx deploy --help
  ```
  The different environments are specified in the .dbx/project.json file. If no environment is specified when call a cli command, the default environment is used.


### Running integration tests
Integration tests are designed to test a project execution as a whole. These type of tests utilize dbx's [asset-based deployment method](https://dbx.readthedocs.io/en/latest/features/assets/#assets-based-workflow-deployment-and-launch) and [--trace parameter](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#launching-the-workflow) to monitor the jobs execution.  

To test the model training workflow, simply run the same commands noted above.
```
dbx deploy ml-prod-deployment-train-model --assets-only
dbx launch ml-prod-deployment-train-model --from-assets --trace
```
If the Job fails for any reason, it will trigger an error in your terminal or CICD server. The failed Job run will contain a copy of the Databricks Notebook and any error messsages.

To test the model inference workflow, run the below commands
```
dbx deploy ml-prod-deployment-score-records-one-time --assets-only
dbx launch ml-prod-deployment-score-records-one-time --from-assets --trace
```
The inference workflow contains an assert statement that can optionally be triggered to test that inference results are as epxected.