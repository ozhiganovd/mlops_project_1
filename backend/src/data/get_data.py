"""
Программа: Получение данных из файла
Версия: 1.0
"""

from typing import Text
from pandas import read_csv, DataFrame

def load_data(train_path: Text) -> DataFrame:
    """
    Загрузка датасета
    :param train_path: адрес месторасположения
    :return: датасет
    """
    return read_csv(train_path, parse_dates=['date'])
