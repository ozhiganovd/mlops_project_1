"""
Программа: Получение метрик
Версия: 1.0
"""

import json

import yaml
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import pandas as pd


def get_dict_metrics(
    y_test: pd.Series, y_predict: pd.Series
) -> dict:
    """
    Создание словаря с метриками
    :param y_test: тестовые данные целевой переменной
    :param y_predict: предсказанные значения
    :return: словарь с метриками
    """
    dict_metrics = {
        'R2': round(r2_score(y_test, y_predict), 3),
        'MAE': round(mean_absolute_error(y_test, y_predict), 3),
        'MSE': round(mean_squared_error(y_test, y_predict), 3)
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: месторасположение сохранения метрик
    """
    result_metrics = get_dict_metrics(
        y_test=data_y,
        y_predict=model.predict(data_x),
    )
    with open(metric_path, 'w') as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Загрузка метрик из файла
    :param config_path: расположение конфигурационного файла
    :return: метрики
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['train']['metrics_path']) as json_file:
        metrics = json.load(json_file)

    return metrics