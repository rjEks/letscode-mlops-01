import joblib
import sklearn
import json
import numpy as np

def handler(event, context):


    sepal_length = float(event.get('Iris')['sepal_length'])
    sepal_width = float(event.get('Iris')['sepal_width'])
    petal_length = float(event.get('Iris')['petal_length'])
    petal_width = float(event.get('Iris')['petal_width'])

    row_arti = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4)

    modelo = joblib.load("modelo.joblib")
    resposta = int(modelo.predict(row_arti)[0])

    return {

        'statusCode': 200,
        'body': resposta
    }