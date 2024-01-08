"""
Программа: Получение предсказания по обученной модели для новых данных
Версия: 1.0
"""

import os
import yaml
import joblib
import pandas as pd
from ..data.get_data import load_data
from ..transform.transform import pipeline

def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Функция предобработки данных и получение предсказаний
    :param dataset: датасет для предсказания
    :param config_path: расположение конфигурационного файла
    :param data_path: расположение файла с данными
    :return: предсказания
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config['preprocessing']
    train_config = config['train']

    if data_path:
        dataset = load_data(train_path=data_path)

    dataset = pipeline(data=dataset, **preprocessing_config)

    model = joblib.load(os.path.join(train_config['model_path']))
    prediction = model.predict(dataset).round().tolist()

    return prediction