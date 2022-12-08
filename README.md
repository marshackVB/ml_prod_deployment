## Deploying and testing model training and inference workflows across Databricks Workspaces 
This repository contains an example machine learning project implementation [using dbx](https://dbx.readthedocs.io/en/latest/). The dbx library is used to create a python project directly structure that is saved in a git servers, such as Azure DevOps or Github and synced to a Databricks Repo. The deployment functionality of the dbx library is then used to deploy jobs across workspace and run integration tests. Using dbx, these operations are easily triggered from the command line, and hence, a CICD server.



### Creating an dbx project via the terminal
```
dbx init -p "cicd_tool=Azure DevOps" \     
         -p "cloud=Azure" \
         -p "project_name=ml-prod-deployment" \
         -p "profile=ml-prod-deployment" \     
        --no-input
```
The profile corresponds to a Databricks CLI profile containing a Workspace URL and PAT token. This sets the default environment in the .dbx/projects.json file. Addition environments can be added to this file making it easy to run jobs in different Workspaces. Running this command creates a default dbx project; it's structure is [described here](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#project-structure). For a quick introduction to dbx project provisioning process, [see the documentation](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#project-structure)


### Running integration tests
An integation tests will tests the projects execution as a whole. These type of tests utile dbx's [asset-based deployment method](https://dbx.readthedocs.io/en/latest/features/assets/#assets-based-workflow-deployment-and-launch) and [--trace parameter](https://dbx.readthedocs.io/en/latest/guides/python/python_quickstart/#launching-the-workflow) to monitor the jobs execution.  

```
dbx deploy ml-prod-deployment-create-features --assets-only
dbx launch ml-prod-deployment-create-features --from-assets --trace
```


Critical tests:
 - Single node cluster
 - Deploy job with schedule
 