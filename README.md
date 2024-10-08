End To End ChatBot mlops project


import dagshub
dagshub.init(repo_owner='RiemanNClav', repo_name='Traffic-Alert-CDMX-Chatbot', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)