"""
Программа: Тренировка модели на backend
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history
import plotly.figure_factory as ff


def start_training(config: dict, endpoint: object) -> None:
    """
    Функция обучения модели с визуализацией
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    else:
        old_metrics = {"R2": 0, "MAE": 0, "MSE": 0}

    with st.spinner('Подбор параметров...'):
        output = requests.post(endpoint, timeout=8000)
    st.success('Успешно!')

    new_metrics = output.json()['metrics']

    R2, MAE, MSE = st.columns(3)
    R2.metric(
        'R2',
        new_metrics['R2'],
        f"{new_metrics['R2']-old_metrics['R2']:.3f}",
    )
    MAE.metric(
        'MAE',
        new_metrics['MAE'],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",
    )
    MSE.metric(
        'MSE',
        new_metrics['MSE'],
        f"{new_metrics['MSE']-old_metrics['MSE']:.3f}",
    )

    study = joblib.load(os.path.join(config['train']['study_path']))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)