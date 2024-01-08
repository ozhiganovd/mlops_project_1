"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st

from src.data.get_data import load_dataset_path, func_load_data
from src.plotting.charts import boxplotting, lineplotting, heatmapplotting
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file
from src.transform.transform import pipeline
import pandas as pd

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """

    st.markdown("# Описание проекта")
    st.title('Прогнозирование кол-ва клиентов на АЗС по дням и часам')
    st.write(
        """
        Есть данные по количеству транзакций для одной АЗС за 2021-2023 года в разрезе дней и часов. 
        Необходима модель для прогнозирования количества транзакций/клиентов на дату по часам для построения 
        графиков персонала под поток покупателей. Вывод оптимального количества сотрудников под трафик позволит 
        увеличить продажи и сократить затраты на персонал.\n
        Задача регрессии.\n
        *Целевая переменная*: count_order.\n
        *Источник данных праздничных, предпраздничных, нерабочих дней:* 
        https://raw.githubusercontent.com/d10xa/holidays-calendar/master/json/calendar.json"""
    )

    st.markdown(
        """
        ### Описание полей 
            - date - дата
            - hour - час
            - holidays - выходной/праздничный день
            - preholidays - предвыходной/предпраздничный день
            - nowork - нерабочий день
            - year - год
            - month - месяц
            - day - день месяца
            - day_name - день недели
            - period_day - период дня:
                1. hours_night - ночь
                2. hours_morning - утро
                3. hours_day - день
                4. hours_evening - вечер 
            - season - сезон года:
                1. winter - зима
                2. spring - весна
                3. summer - лето
                4. autumn - осень
    """
    )


def eda():
    """
    Разведочный анализ данных
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config['preprocessing']

    data = load_dataset_path(dataset_path=preprocessing_config['train_path'])
    pipeline(data=data, **preprocessing_config)
    st.write(data.head())

    year_orders = st.sidebar.checkbox('Транзакции по годам')
    weekday_orders = st.sidebar.checkbox("Транзакции по дням недели")
    work_orders = st.sidebar.checkbox("Транзакции в рабочие и нерабочие дни")
    preholidays_orders = st.sidebar.checkbox("Транзакции в предпраздничные дни")
    period_day_orders = st.sidebar.checkbox("Транзакции по временам суток")
    linear_dependence_analysis = st.sidebar.checkbox("Анализ линейной зависимости")

    if year_orders:
        st.pyplot(
            boxplotting(
                data=data.groupby(['year', 'date'], as_index=False).agg({'count_order': 'sum'}),
                data_x='count_order',
                hue='year',
                title='Среднесуточное кол-во транзакций по годам',
                xlabel='Кол-во транзакций в стуки'
            )
        )
        st.write('Сильного различия в количестве клиентов за 3 года нет. '
                 'В 2022 году более низкикие показатели основных статитстик - связано с началом сво и мобилизацией.')

    if weekday_orders:
        st.pyplot(
            boxplotting(
                data=data.groupby(['date', 'day_name'], as_index=False).agg({'count_order': 'mean'}),
                data_x='count_order',
                hue='day_name',
                title='Среднесуточное кол-во транзакций по дням недели',
                xlabel='Кол-во транзакций в стуки'
            )
        )
        st.write('Кол-во клиентов зависит от дня недели: пик продаж приходится на пятницу, '
                 'самые низкие продажи в воскресенье и субботу')
    if work_orders:
        data['work'] = data.iloc[:, 7:9].sum(axis=1) * -1 + 1
        st.pyplot(
            boxplotting(
                data=pd.melt(data.groupby(['date', 'year', 'holidays', 'preholidays', 'nowork', 'work'], as_index=False).agg({'count_order': 'sum'}), id_vars=['year', 'date', 'count_order']).query('value == 1'),
                data_x='count_order',
                hue='variable',
                title='Среднесуточное кол-во транзакций в рабочие, нерабочие, праздничные и предпраздничные дни',
                xlabel='Кол-во транзакций в сутки'
            )
        )
        st.write('Больше всего клиентов АЗС обслуживает в предпраздничные и рабочие дни. '
                 'В праздничные и нерабочие дни клиентский поток снижается.')

    if preholidays_orders:
        st.pyplot(
            boxplotting(
                data=data.query('preholidays == 1').groupby(['year', 'date', 'period_day'], as_index=False).agg({'count_order': 'sum'}),
                data_x='count_order',
                hue='period_day',
                title='Среднесуточное кол-во транзакций в предпраздничные дни',
                xlabel='Кол-во транзакций в сутки'
            )
        )
        st.write('В предпраздничные дни основной поток клиентов приходится на вечерние часы, '
                 'наименьшее кол-во клиентов в ночные часы.')

    if period_day_orders:
        st.pyplot(
            lineplotting(
                data=data,
                data_x='hour',
                data_y='count_order',
                title='Среднесуточное кол-во транзакций по часам',
                xlabel='Часы'
            )
        )
        st.write('В ночные часы с 0 до 4 утра минимальное кол-во клиентов. '
                 'Пик продаж приходится на вечерние часы.')

    if linear_dependence_analysis:
        st.pyplot(
            heatmapplotting(
                data=data
            )
        )
        st.write('1. Между целевой переменной и признаками отсутствует линейная зависимость;\n'
                 '2. Обратная линейная зависимость между выходными и рабочими днями.')


def training():
    """
    Функция обучения модели
    """
    st.markdown('# Обучение модели LightGBM')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['train']

    if st.button('Начать обучение'):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний по введённым данным
    """
    st.markdown('# Предсказание по введённым данным')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_input']
    uniq_features_path = config['preprocessing']['uniq_features_path']

    if os.path.exists(config['train']['model_path']):
        evaluate_input(unique_data_path=uniq_features_path, endpoint=endpoint)
    else:
        st.error('Модель не обучена')


def prediction_from_file():
    """
    Получение предсказаний по данным из файла
    """
    st.markdown('# Предсказание по файлу')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_from_file']

    upload_file = st.file_uploader(
        '', type=['csv'], accept_multiple_files=False
    )
    if upload_file:
        dataset_csv_df, files = func_load_data(data=upload_file, type_data='Test')
        if os.path.exists(config['train']['model_path']):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error('Модель не обучена')


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        'Описание проекта': main_page,
        'Exploratory data analysis': eda,
        'Обучение модели': training,
        'Предсказание по введённым данным': prediction,
        'Предсказание по файлу': prediction_from_file,
    }
    selected_page = st.sidebar.selectbox('Выберите раздел', page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == '__main__':
    main()