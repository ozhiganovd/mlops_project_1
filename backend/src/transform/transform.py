"""
Программа: Предобработка
тренировочных и тестовых данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def uniq_features(
        data: pd.DataFrame, drop_col: list, target_col: str, uniq_features_path: str) -> None:
    """
    Функция для сохранение словаря с признаками и уникальными значениями
    :param data: подаваемый датафрейм
    :param drop_col: список с названиями признаков для удаления
    :param target_col: целевая переменная
    :param uniq_features_path: путь для сохранения
    :return: None
    """
    unique_df = data.drop(columns=drop_col + [target_col], axis=1, errors='ignore')
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(uniq_features_path, "w") as file:
        json.dump(dict_unique, file)


def map_values(data: pd.DataFrame, map_columns: dict) -> pd.DataFrame:
    """
    функция для замены значений в колонках
    :param data: подаваемый DataFrame
    :param map_columns: словарь с заменяемыми значениями
    :return: датафрейм
    """
    return data.replace(map_columns)


def astype_col(data: pd.DataFrame, map_type_columns: dict) -> pd.DataFrame:
    """
    функция для замены типов данных в колонках
    :param data: подаваемый DataFrame
    :param map_type_columns: словарь с колонками и типом данных
    :return: датафрейм
    """
    return data.astype(map_type_columns)


def check_columns(data: pd.DataFrame, drop_col: list, target_col: str, values_path: str) -> pd.DataFrame:
    """
    Проверка признаков на соответствие train
    :param data: датафрейм test
    :param unique_values_path: путь к списку с признаками
    :return: датафрейм
    """
    with open(values_path) as json_file:
        unique_values = json.load(json_file)

    column = unique_values.keys()
    data_col = data.drop(columns=drop_col + [target_col], axis=1, errors='ignore')

    assert set(column) == set(data_col.columns), 'Не соответствуют признаки'
    return data


def pipeline(data: pd.DataFrame, flag_eval: bool = True, **kwargs):
    """
    Функция предобработки данных
    :param data: подаваемый датафрейм
    :param flag_eval: флаг датфрейма - для обучения или для предсказания
    :return: датафрейм для обучения/предсказания
    """
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    data['month'] = data.month.map(kwargs['month'])
    data['day'] = data.date.dt.day
    if flag_eval:
        data = check_columns(
            data=data,
            drop_col=kwargs['drop_columns'],
            target_col=kwargs['target_column'],
            values_path=kwargs['uniq_features_path']
        )
    else:
        uniq_features(
            data=data,
            drop_col=kwargs['drop_columns'],
            target_col=kwargs['target_column'],
            uniq_features_path=kwargs['uniq_features_path']
        )
    data['day_name'] = data.date.dt.day_name()
    js = json.load(open(kwargs['holidays_calendar']))
    for x in js:
        data[x] = data['date'].isin(pd.to_datetime(pd.DataFrame(js[x], columns=['date']).date, format='%Y-%m-%d'))
    data = astype_col(data=data, map_type_columns=kwargs['map_type_col'])
    data['period_day'] = data.hour.map(kwargs['period_day'])
    data['season'] = data.month.map(kwargs['season'])
    data = data.drop(kwargs['drop_columns'], axis=1, errors='ignore')

    cat_cols = data.select_dtypes('object').columns
    data[cat_cols] = data[cat_cols].astype('category')

    return data