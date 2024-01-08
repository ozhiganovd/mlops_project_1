"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(dataset: pd.DataFrame, **kwargs):
    """
    Разделение датасета на train/test с последующим сохранением
    :param dataset: датасет
    :return: train/test датасеты
    """
    df_train, df_test = train_test_split(
        dataset,
        test_size=kwargs['test_size'],
        random_state=kwargs['random_state'],
    )
    df_train.to_csv(kwargs['train_path_proc'], index=False)
    df_test.to_csv(kwargs['test_path_proc'], index=False)
    return df_train, df_test


def train_test_data(data_train: pd.DataFrame, data_test: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых на объект-признаки и целевую переменную
    :param data_train: train данные
    :param data_test: test данные
    :param target: название целевой переменной
    :return: набор данных train/test
    """
    x_train, x_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )
    return x_train, x_test, y_train, y_test
