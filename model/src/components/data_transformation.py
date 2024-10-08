import sys
from dataclasses import dataclass, field

import numpy as np 
import pandas as pd
import mlflow

from src.exception import CustomException
from src.logger import logging
import os


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder,  OrdinalEncoder

from src.utils import save_object

@dataclass
class ConstantsConfig:
    variables_categoricas: list = field(default_factory=lambda: ["mes", 'categoria',
                                                                 "color_aux", "marca_general",
                                                                "colonia", "alcaldia"])
    variables_numericas: list = field(default_factory=lambda: ["anio_infraccion"])

class DataCleaning(object):
    def __init__(self):
        pass

    def clasificar_categoria(self, x):
        if isinstance(x, str):
            if "INCOMPLETA" in x or "NO TENER LICENCIA" in x:
                return "DOCUMENTACION INCOMPLETA"
            elif "MAL USO" in x or "ABANDONO" in x or "ESTACIONAR" in x or "VIA" in x or "INFRINGIR" in x:
                return "USO INCORRECTO DE LA VIA PUBLICA"
            elif "EXCEDER" in x:
                return "EXCESO DE VELOCIDAD"
            elif "EFECTOS" in x:
                return "CONDUCIR BAJO INFLUENCIA"
            elif "NO RESPETAR" in x or "VIOLACIONES" in x or "INMOVILIZACION" in x or "VEHICULOS" in x:
                return "INFRACCIONES DE TRANSITO"
            elif "INCUMPLIR" in x or "VIOLAR NORMAS" in x or "PONER EN RIESGO" in x:
                return "SEGURIDAD VIAL"
        return x

    def initiate_data_cleaning(self, df):

        try:

            df = pd.read_csv(df)

            #train_df = train_path
            #test_df = test_path

            # reemplazo y traducción para caracteres especiales
            trans_dict = str.maketrans('ÁÉÍÓÚ', 'AEIOU')
            df['anio_infraccion'] = df['anio_infraccion'].astype(int)
            df['mes'] = df['mes'].str.upper()
            df['categoria'] = df['categoria'].str.upper().replace(r"[.,©@#³$%&/()=!¡\?¿]", "", regex=True).str.translate(trans_dict)
            df['color_aux'] = df['Color'].str.upper().replace(r"[.,©@#³$%&/()=!¡\?¿]", "", regex=True).str.translate(trans_dict)
            df['marca_general'] = df['marca_general'].str.upper().replace(r"[.,©@#³$%&/()=!¡\?¿]", "", regex=True).str.translate(trans_dict)
            df['colonia'] = df['colonia'].str.upper().replace(r"[.,©@#³$%&/()=!¡\?¿]", "", regex=True).str.translate(trans_dict)
            df['alcaldia'] = df['alcaldia'].str.upper().replace(r"[.,©@#³$%&/()=!¡\?¿]", "", regex=True).str.translate(trans_dict)
            df['fecha_infraccion'] = pd.to_datetime(df['fecha_infraccion'])
            df['dia'] = df['fecha_infraccion'].apply(lambda r:r.day).astype('int')

            df['alcaldia'] = df['alcaldia'].apply(lambda x: "AZCAPOTZALCO" if isinstance(x, str) and "ALCO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "XOCHIMILCO" if isinstance(x, str) and "XO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "LA MAGDALENA CONTRERAS" if isinstance(x, str) and "MAGDA" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "COYOACAN" if isinstance(x, str) and "COYO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "TLAHUAC" if isinstance(x, str) and "TLAL" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "BENITO JUAREZ" if isinstance(x, str) and "CUAJ" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "MILPA ALTA" if isinstance(x, str) and "MILPA" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "IZTAPALAPA" if isinstance(x, str) and "TAPALA" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "IZTACALCO" if isinstance(x, str) and "CALCO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "CUAUHTEMOC" if isinstance(x, str) and (("MOC" in x) or ("CONDESA" in x) or ("ROMA" in x)) else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "GUSTAVO A MADERO" if isinstance(x, str) and "MADERO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "VENUSTIANO CARRANZA" if isinstance(x, str) and "VENUSTIANO" in x else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "MIGUEL HIDALGO" if isinstance(x, str) and (("MIGUEL" in x) or ("%POLANCO" in x)) else x)
            df['alcaldia'] = df['alcaldia'].apply(lambda x: "ALVARO OBREGON" if isinstance(x, str) and "ALVARO" in x else x)


            df['categoria'] = df['categoria'].apply(self.clasificar_categoria)


            # df['categoria'] = df['categoria'].apply(lambda x: "DOCUMENTACION INCOMPLETA" if isinstance(x, str) and "INCOMPLETA" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "DOCUMENTACION INCOMPLETA" if isinstance(x, str) and "NO TENER LICENCIA" in x else x)

            # df['categoria'] = df['categoria'].apply(lambda x: "USO INCORRECTO DE LA VIA PUBLICA" if isinstance(x, str) and "MAL USO" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "USO INCORRECTO DE LA VIA PUBLICA" if isinstance(x, str) and "ABANDONO" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "USO INCORRECTO DE LA VIA PUBLICA" if isinstance(x, str) and "ESTACIONAR" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "USO INCORRECTO DE LA VIA PUBLICA" if isinstance(x, str) and "VIA" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "USO INCORRECTO DE LA VIA PUBLICA" if isinstance(x, str) and "INFRINGIR" in x else x)

            # df['categoria'] = df['categoria'].apply(lambda x: "EXCESO DE VELOCIDAD" if isinstance(x, str) and "EXCEDER" in x else x)

            # df['categoria'] = df['categoria'].apply(lambda x: "CONDUCIR BAJO INFLUENCIA" if isinstance(x, str) and "EFECTOS" in x else x)

            # df['categoria'] = df['categoria'].apply(lambda x: "INFRACCIONES DE TRANSITO" if isinstance(x, str) and "NO RESPETAR" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "INFRACCIONES DE TRANSITO" if isinstance(x, str) and "VIOLACIONES" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "INFRACCIONES DE TRANSITO" if isinstance(x, str) and "INMOVILIZACION" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "INFRACCIONES DE TRANSITO" if isinstance(x, str) and "VEHICULOS" in x else x)



            # df['categoria'] = df['categoria'].apply(lambda x: "SEGURIDAD VIAL" if isinstance(x, str) and "INCUMPLIR" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "SEGURIDAD VIAL" if isinstance(x, str) and "VIOLAR NORMAS" in x else x)
            # df['categoria'] = df['categoria'].apply(lambda x: "SEGURIDAD VIAL" if isinstance(x, str) and "PONER EN RIESGO" in x else x)


            df = df[['anio_infraccion' ,'mes', 'color_aux', 'marca_general', 'colonia', 'dia', 'alcaldia', 'categoria']]
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, categoricas):
        self.frequency_dict = {}
        self.categoricas =   categoricas

    def fit(self, X, y=None):
        # Calcular la frecuencia de cada valor categórico
        for col in self.categoricas:
            values = X[col].value_counts(normalize=True).to_dict()
            self.frequency_dict[col] = values
            
        return self

    def transform(self, X):
        # Reemplazar los valores por su frecuencia
        for col in self.categoricas:

            X[col] = X[col].map(self.frequency_dict[col]).fillna(0).values.reshape(-1, 1)

        return X

        
class SimpleImputer_(BaseEstimator, TransformerMixin):
    
    def __init__(self, categoricas):
        self.imputer = None
        self.categoricas =   categoricas

    def fit(self, X, y=None):
        simple_imputer = SimpleImputer(strategy='most_frequent')
        self.imputer = simple_imputer.fit(X)

            
        return self

    def transform(self, X):
        # Reemplazar los valores por su frecuencia

        X_array= self.imputer.transform(X)
        
        X_df = pd.DataFrame(X_array, columns=self.imputer.get_feature_names_out())


        return X_df


@dataclass
class DataTransformationConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessor_obj_file_path: str = os.path.join(base_dir, 'artifacts', "preprocessor.pkl")
    categorias_file_path: str = os.path.join(base_dir, 'artifacts', "categorias.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            
            numerical_columns = ["anio_infraccion", "dia"]
            categorical_columns = [
                'mes',
                'marca_general',
                'colonia',
                'alcaldia'
            ]
            y = ['categoria']

            # num_pipeline = Pipeline(
            #     steps=[
            #         ("scaler",StandardScaler())
            #     ]
            # )
   
            cat_pipeline_X = Pipeline(
                steps=[
                    ("simple_imputer", SimpleImputer_(categoricas=categorical_columns)),
                    ('frequency_encoder', FrequencyEncoder(categoricas=categorical_columns))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            # preprocessor_X = ColumnTransformer(
            #     [
            #         ("cat_pipeline", cat_pipeline_X, categorical_columns)
                    
            #     ]
            # )

            cat_pipeline_y = Pipeline(
                steps=[
                    ("simple_imputer", SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder())
                ]
            )

            preprocessor_y = ColumnTransformer(
                [
                    ("cat_pipeline", cat_pipeline_y, y)
                    
                ]
            )

            return cat_pipeline_X, preprocessor_y
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= train_path
            test_df= test_path

            categoria = train_df[['categoria']].copy()


            preprocessing_obj_X, preprocessing_obj_y =self.get_data_transformer_object()

            logging.info(f"Read train and test data completed")

            #TRANSFORMAMOS y

            target_column_name="categoria"

            y_train = train_df[[target_column_name]]
            y_test = test_df[[target_column_name]]


            y_train=preprocessing_obj_y.fit_transform(y_train).flatten()

            y_train = y_train.astype(int).astype(str)

            y_test=preprocessing_obj_y.transform(y_test).flatten()

            y_test = y_test.astype(int).astype(str)

            # TRANSFORMAMOS X

            input_feature_train_df=train_df.drop(columns=[target_column_name, 'color_aux', 'anio_infraccion'],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column_name, 'color_aux', 'anio_infraccion'],axis=1)

            input_feature = preprocessing_obj_X.fit(input_feature_train_df)
            input_feature_train_arr=input_feature.transform(input_feature_train_df)
            input_feature_test_arr=input_feature.transform(input_feature_test_df)


            train_arr = input_feature_train_arr.copy()
            test_arr = input_feature_test_arr.copy()
            train_arr['categoria'] = y_train
            test_arr['categoria'] = y_test

            categoria['label'] = y_train 
            categoria = categoria.dropna()
            categoria = categoria.drop_duplicates(['label'])
            categoria.to_csv(self.data_transformation_config.categorias_file_path, index=False)


            train_arr = np.array(train_arr)
            test_arr = np.array(test_arr)



            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj_X

            )

            ## INICIO MLFLOW


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    
    cleaning = DataCleaning()
    train_data  = cleaning.initiate_data_cleaning(train_data)
    test_data  = cleaning.initiate_data_cleaning(test_data)