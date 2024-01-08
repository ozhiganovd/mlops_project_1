"""
Программа: Получение данных
Версия: 1.0
"""

from io import BytesIO
import io
from typing import Dict, Tuple
import streamlit as st
import pandas as pd


def load_dataset_path(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датафрейма по выбранному пути
    :param dataset_path: месторасположение датафрейма
    :return: датафрейм
    """
    return pd.read_csv(dataset_path, parse_dates=['date'])


def func_load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO
    :param data: данные
    :param type_data: тип датафрейма train df/test df
    :return: датафрейм, датафрейм в формате BytesIO
    """
    dataset = pd.read_csv(data, parse_dates=['date'])
    st.write("Dataset load")
    st.write(dataset.head())
    dataset_bytes_obj = io.BytesIO()
    dataset.to_csv(dataset_bytes_obj, index=False)
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files