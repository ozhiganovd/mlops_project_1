"""
Программа: Модель прогнозирования кол-ва
клиентов на АЗС по часам и дням
Версия: 1.0
"""

import warnings
import optuna
import uvicorn
import pandas as pd

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import train_pipeline
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = '../config/params.yml'


class GasStationCustomers(BaseModel):
    """
    Признаки для получения результатов модели
    """
    date: str
    hour: int


@app.post('/train')
def training():
    """
    Функция обучения модели
    """
    train_pipeline(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {'metrics': metrics}


@app.post('/predict')
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), 'Тип результата не соответствует list'
    return {'prediction': result[:5]}


@app.post('/predict_input')
def prediction_input(customer: GasStationCustomers):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            customer.date,
            customer.hour
        ]
    ]

    cols = [
        'date',
        'hour'
    ]

    data = pd.DataFrame(features, columns=cols)
    data['date'] = pd.to_datetime(data.date, format='%Y-%m-%d')
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    return predictions


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)