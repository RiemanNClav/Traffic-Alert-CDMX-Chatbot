import requests


def send_data_to_model(data):
    url = 'http://model:5001/predict'  # URL del servicio de modelo
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['predictions']
    else:
        return None
    

if __name__=="__main__":

    input_data = {
        'mes': ['SEPTIEMBRE'],
        'marca_general': ['CHEVY'],
        'colonia': ['POLANCO'],
        'dia': [15],
        'alcaldia': ['MIGUEL HIDALGO']
    }

    predictions = send_data_to_model(input_data)
    print(predictions)