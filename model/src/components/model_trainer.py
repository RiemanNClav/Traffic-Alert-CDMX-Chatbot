import os
import sys
from dataclasses import dataclass
import numpy as np

# modelos
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report



#metricas, parametros
from sklearn.model_selection import  GridSearchCV


#mlflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
#from src.components.models import models_params

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def models_params(self):
        models_params_ = {
            "Multinomial Random Forest": {
                "model": RandomForestClassifier(),
            "params": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [5,10, 20]
                    }
                }
        }
        return models_params_
        


    def initiate_model_trainer(self,train_array,test_array):
        try:
            experiment_name = "Modelo de Clasificacion"
            artifact_repository = './mflow-run'

            mlflow.set_tracking_uri('http://127.0.0.1:5000/')
            #Inicializar cliente MLFLOW
            cliente = MlflowClient()
    
            try:
                experiment_id = cliente.create_experiment(experiment_name, artifact_location=artifact_repository)
            except:
                experiment_id = cliente.get_experiment_by_name(experiment_name).experiment_id

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Entrena y registra los modelos
            report_r2 = {}
            report_params = {}

            for model_name, config in self.models_params().items():
                model = config["model"]
                params = config["params"]

                gs = GridSearchCV(model, params)
            
                # Inicia una nueva ejecución en MLflow
                with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:

                        # Obtener identificación de ejecución
                    run_id = run.info.run_uuid
        
                     # Proporcione notas breves sobre el run.
                    MlflowClient().set_tag(run_id,
                                           "mlflow.note.content",
                                           "Este es un experimento para explorar diferentes modelos de aprendizaje automático para Campus Recruitment Dataset")
                    mlflow.sklearn.autolog()

                    # Definimos el custom tag
                    tags = {"Application": "CDMX traffic alert",
                            "release.candidate": "PMP",
                            "release.version": "2.2.0"}
                    
                    # Set Tag
                    mlflow.set_tags(tags)
                                    
                    # Log python environment details
                    #mlflow.log_artifact('PizzaPredictionV2/requirements.txt')

                    gs.fit(X_train, y_train)
                    #mlflow.log_params(gs.best_params_) #registro mejores parametros
                    mlflow.sklearn.log_model(gs.best_estimator_, "model") # registra el modelo

                    # Evalúa el modelo en el conjunto de test y registra la métrica
                    model.set_params(**gs.best_params_)
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)

                    # evalua metricas
                    accuracy  = accuracy_score(y_test, y_test_pred)
                    precision_multi  = precision_score(y_test, y_test_pred, average='weighted')
                    recall_multi  = recall_score(y_test, y_test_pred, average='weighted')
                    f1_multi  = f1_score(y_test, y_test_pred, average='weighted')
                    #roc_auc = roc_auc_score(y_test, y_test_pred)


                    # registro de metricas
                    mlflow.log_metric("accuracy", accuracy)
                    print(f'accuracy: {accuracy}')
                    mlflow.log_metric("precision", precision_multi)
                    print(f'precision: {precision_multi}')

                    mlflow.log_metric("recall", recall_multi)
                    print(f'recall: {recall_multi}')

                    mlflow.log_metric("F1", f1_multi)
                    print(f'F1: {f1_multi}')

                    # mlflow.log_metric("ROC", roc_auc)
                    # print(f'ROC: {roc_auc}')


                    mlflow.sklearn.log_model(model, model_name)

                    report_r2[model_name] = accuracy
                    report_params[model_name] = gs.best_params_

            # mejor accuracy
            best_model_score = max(sorted(report_r2.values()))

            #mejor modelo
            best_model_name = list(report_r2.keys())[
                list(report_r2.values()).index(best_model_score)
                ]
            #mejores parametros
            best_params = report_params[best_model_name]


            if best_model_score<0.6:

                raise CustomException("No best model found")
            
            else:

                logging.info(f"Best found model on both training and testing dataset")

            best_model_obj = self.models_params()[best_model_name]["model"]

            best_model_obj.set_params(**best_params)

            best_model_obj.fit(X_train, y_train)

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model_obj)
                            
        except Exception as e:
            raise CustomException(e,sys)