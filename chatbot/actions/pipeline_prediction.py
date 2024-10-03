import os
import sys
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import dill


class PredictPipeline:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    def load_model_and_preprocessor(model_name, preprocessor_path):

        model_uri = f"models:/{model_name}/latest"  
        loaded_model = mlflow.sklearn.load_model(model_uri)
        preprocessor = joblib.load(preprocessor_path)
        return loaded_model, preprocessor
    


    def fill_model_cache():
        global MODELS
        client = MlflowClient()
        registered_models = client.search_registered_models(
        filter_string="tags.prediction of traffic incidents='true'")
        
        for registered_model in registered_models:
            try:
                model_version = client.get_model_version_by_alias(registered_model.name, "champion")
                model = mlflow.sklearn.load_model(f"models:/{registered_model.name}@champion")
                model_name, *_ = registered_model.name.split("__")
                MODELS[model_name] = (model_version, model)
                logger.info("model loaded successfully", extra={"model": model_version.name})


                temp_dir = "temp_artifacts"  
                os.makedirs(temp_dir, exist_ok=True)
                preprocessor_path = client.download_artifacts(run_id, "preprocessor.pkl", temp_dir)

                with open(preprocessor_path, 'rb') as f:
                    preprocessor = dill.load(f)




                

            except Exception as e:
                logger.error("model failed to load", extra={"model": registered_model.name, "error": str(e)})






    def predict(self,features):
        preprocessor_path=os.path.join(self.base_dir, "artifacts","preprocessor.pkl")
        model_path=os.path.join(self.base_dir, "artifacts","model.pkl")
        categorias_path = os.path.join(self.base_dir, "artifacts","categorias.csv")
            
        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)


        categorias = pd.read_csv(categorias_path)
        dicc = dict(zip(list(categorias.label), list(categorias.categoria)))
            
        scaled_data=preprocessor.transform(features)
            
        pred_label = model.predict(scaled_data)[0]

        pred = dicc[int(pred_label)]


        return pred
            
    
class CustomData:
    def __init__(self,
                 mes:str,
                 dia:int,
                 marca_general:str,
                 colonia:str,
                 alcaldia:str):
        
        self.mes=mes
        self.marca_general=marca_general.upper()
        self.colonia=colonia.upper()
        self.dia=dia
        self.alcaldia=alcaldia.upper()
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'mes':[self.mes],
                    'marca_general':[self.marca_general],
                    'colonia':[self.colonia],
                    'dia':[self.dia],
                    'alcaldia':[self.alcaldia]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)
            

if __name__=="__main__":
    data = CustomData('SEPTIEMBRE',
                      29,
                      'CHEVY',
                      'TACUBAYA',
                      "MIGUEL HIDALGO")


    pred_df = data.get_data_as_dataframe()


    predict_pipeline=PredictPipeline()

    results=predict_pipeline.predict(pred_df)

    print(f'Prediction = {results}')