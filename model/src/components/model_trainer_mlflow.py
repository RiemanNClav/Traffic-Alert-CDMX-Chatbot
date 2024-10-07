import os
import sys
from dataclasses import dataclass
import numpy as np
import dill

# modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
#metricas, parametros
from sklearn.model_selection import  GridSearchCV

#mlflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
#from src.components.models import models_params

@dataclass
class ModelTrainerConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trained_model_file_path=os.path.join(base_dir, "artifacts", "model.pkl")
    preprocessing_file_path=os.path.join(base_dir, "artifacts", "preprocessor.pkl")

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
                    'n_estimators': [5]
                    }
                }
        }
        return models_params_
        


    def initiate_model_trainer(self,train_array,test_array, run_id_):
        try:

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
            report_metrics = {}

            # ENTRENAMIENTO DE DISTINTOS MODELOS

            for model_name, config in self.models_params().items():
                model = config["model"]
                params = config["params"]

                gs = GridSearchCV(model, params)

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                # evalua metricas
                accuracy  = accuracy_score(y_test, y_test_pred)
                precision_multi  = precision_score(y_test, y_test_pred, average='weighted')
                recall_multi  = recall_score(y_test, y_test_pred, average='weighted')
                f1_multi  = f1_score(y_test, y_test_pred, average='weighted')
                #roc_auc = roc_auc_score(y_test, y_test_pred)

                report_r2[model_name] = accuracy
                report_params[model_name] = gs.best_params_
                report_metrics[model_name] = {'accuracy': accuracy, 'precision': precision_multi,
                                              'recall': recall_multi, 'F1': f1_multi}


             # LOGICA PARA ENCONTRAR EL MEJOR MODELO   

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

                logging.info(f"It has found a champion model")


            best_model_obj = self.models_params()[best_model_name]["model"]

            best_model_obj.set_params(**best_params)

            best_model_obj.fit(X_train, y_train)

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model_obj)
            

            # REGISTRO EN MLFLOW 

            # MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")

            with mlflow.start_run(run_id=run_id_):
                client = MlflowClient()

                # mlflow.log_param(f"X_train", X_train.shape)
                # mlflow.log_param(f"y_train", y_train.shape)
                # mlflow.log_param(f"X_test", X_test.shape)
                # mlflow.log_param(f"y_test", y_test.shape)

                mlflow.sklearn.log_model(best_model_obj, f"{best_model_name}/model")

                version = mlflow.register_model(f"runs:/{run_id_}/{best_model_name}/model", f"{best_model_name}__model")
                client.set_registered_model_alias(
                    version.name,
                    "champion",  # Make this the champion model for now
                    version.version,)

                client.set_registered_model_tag(
                    version.name,
                    "prediction_incidents",
                    "true",)
                
                print(best_model_name)
                print('--------------------------------------')
                print('\n')

                accuracy = report_metrics[best_model_name]['accuracy']
                mlflow.log_metric("accuracy",accuracy)
                print(f'accuracy: {accuracy}')

                precision_multi = report_metrics[best_model_name]['precision']
                mlflow.log_metric("precision", precision_multi)
                print(f'precision: {precision_multi}')

                recall_multi = report_metrics[best_model_name]['recall']
                mlflow.log_metric("recall", recall_multi)
                print(f'recall: {recall_multi}')

                f1_multi = report_metrics[best_model_name]['F1']
                mlflow.log_metric("F1", f1_multi)
                print(f'F1: {f1_multi}')

                print('\n')
                print('---------------------------------------')

            # ## GUARDAR ARTEFACTOS ADICIONALES
            #     preprocessor = load_object(self.model_trainer_config.preprocessing_file_path)
            #     pipeline_filename = "preprocessor.pkl"
                
            #     with open(pipeline_filename, 'wb') as f:
            #         dill.dump(preprocessor, f)

            #     # Registra el archivo en MLflow
            #     mlflow.log_artifact(pipeline_filename)

                            
        except Exception as e:
            raise CustomException(e,sys)