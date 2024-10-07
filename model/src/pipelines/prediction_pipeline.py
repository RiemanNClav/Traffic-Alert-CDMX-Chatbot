import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def predict(self,features):
        try:
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
            
            
        
        except Exception as e:
            raise CustomException(e,sys)
    
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