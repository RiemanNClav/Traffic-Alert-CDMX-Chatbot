import os
import sys
import Levenshtein
from pandas import DataFrame
import pandas as pd
import dill

from src.exception import CustomException


def detect_nulls(df: DataFrame, threshold: float, squema: list):

    try:
        df = df[squema]
        df = df.select(squema)
        null_percentage = (df.isnull().sum() / len(df))

        filtered_columns = null_percentage[null_percentage <= threshold].index.tolist()
        return filtered_columns
    
    except Exception as e:
        raise CustomException(e,sys)
    


def categoria_mas_parecida(cadena, lista_categorias):
    categoria_parecida = None
    distancia_minima = float('inf')
    
    for categoria in lista_categorias:
        distancia = Levenshtein.distance(cadena, categoria)
        
        if distancia < distancia_minima:
            distancia_minima = distancia
            categoria_parecida = categoria
            
    return categoria_parecida


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    


        
if __name__=="__main__":

    list =  ['id_infraccion', 'fecha_infraccion', 'ao_infraccion', 'mes', 'categoria', 
                    'articulo', 'fraccion', 'inciso', 'parrafo', 'placa', 
                    'Color', 'marca_general', 'marca', 'submarca', 'en_la_calle', 
                    'entre_calle', 'y_calle', 'colonia', 'alcaldia', 'longitud', 'latitud']
    
    x = categoria_mas_parecida('color', list)

    print(x)