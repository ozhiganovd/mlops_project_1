"""
Программа предобработки данных
Версия: 2.0
"""

import json
import pandas as pd

def astype_col(data: pd.DataFrame, map_type_columns: dict) -> pd.DataFrame:
    """
    функция для замены типов данных в колонках
    :param data: подаваемый DataFrame
    :param map_type_columns: словарь с колонками и типом данных
    :return: датафрейм
    """
    return data.astype(map_type_columns, errors="raise")

def pipeline(data: pd.DataFrame, **kwargs):
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
    data['day_name'] = data.date.dt.day_name()
    js = json.load(open(kwargs['holidays_calendar']))
    for x in js:
        data[x] = data['date'].isin(pd.to_datetime(pd.DataFrame(js[x], columns=['date']).date, format='%Y-%m-%d'))
    data['period_day'] = data.hour.map(kwargs['period_day'])
    data['season'] = data.month.map(kwargs['season'])
    data = astype_col(data=data, map_type_columns=kwargs['map_type_col'])

    return data