"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
import datetime


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Функция получения данных через ручной ввод
    :param unique_data_path: месторасположение уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    date = st.sidebar.date_input('date', min_value=datetime.date(2023, 1, 1))
    hour = st.sidebar.slider('hour', min_value=0, max_value=23)

    dict_data = {
        'date': str(date),
        'hour': hour
    }

    st.write(
        f"""### Данные АЗС:\n
    1) date: {dict_data['date']}
    2) hour: {dict_data['hour']}
    """
    )

    button_ok = st.button('Применить модель')
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {int(output)}")
        st.success('Успешно!')


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Функция чтения данных из файла и вывода в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button('Применить модель')
    if button_ok:
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_['count_orders'] = output.json()['prediction']
        st.write(data_.head())