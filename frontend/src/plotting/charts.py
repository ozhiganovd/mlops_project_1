"""
Программа: Визуализация
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def boxplotting(
    data: pd.DataFrame, data_x: str, hue: str, title: str, xlabel: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: подаваемый датафрейм
    :param data_x: данные для оси OX
    :param hue: группирвока по признаку
    :param title: название графика
    :xlabel: подпись оси OX
    :return: поле графика
    """

    fig = plt.figure(figsize=(15, 7))

    sns.boxplot(
        data=data, x=data_x, hue=hue, palette="flare"
    )
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def lineplotting(
        data: pd.DataFrame, data_x: str, data_y: str, title: str, xlabel: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика lineplot
    :param data: подаваемый датафрейм
    :param x: данные для оси OX
    :param y: данные для оси OY
    :param title: название графика
    :param xlabel: подпись оси OX
    :return: поле графика
    """
    fig = plt.figure(figsize=(15, 7))
    sns.lineplot(data=data, x=data_x, y=data_y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def heatmapplotting(
    data: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отрисовка графика heatmap
    :param data: подаваемый датафрейм
    :return: поле графика
    """
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(15, 7))

    sns.heatmap(
        data=data[['hour', 'count_order', 'holidays', 'preholidays', 'nowork', 'year', 'day', 'work']].corr(method='spearman'), annot=True, fmt='.2f', cmap="crest")
    plt.title('Анализ линейной зависимости', fontsize=20)
    return fig