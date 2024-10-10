from flask import Flask, request, jsonify
import pandas as pd
import os
from src.utils import load_object
# from src.pipelines.prediction_pipeline import CustomData

app = Flask(__name__)

# Cargar el preprocesador y el modelo al iniciar la aplicación
base_dir = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.join(base_dir, "artifacts", "preprocessor.pkl")
model_path = os.path.join(base_dir, "artifacts", "model.pkl")
categorias_path = os.path.join(base_dir, "artifacts","categorias.csv")

preprocessor = load_object(preprocessor_path)
model = load_object(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    scaled_data = preprocessor.transform(df)
    pred_label = model.predict(scaled_data)[0]

    #categorias.csv
    categorias = pd.read_csv(categorias_path)
    dicc = dict(zip(list(categorias.label), list(categorias.categoria)))

    pred = dicc[int(pred_label)]
    
    return jsonify({'predictions': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)