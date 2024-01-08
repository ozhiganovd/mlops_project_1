"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import os
import joblib
import yaml

from ..data.split_dataset import split_dataset
from ..train.train import optimal_params, train_model
from ..data.get_data import load_data
from ..transform.transform import pipeline


def train_pipeline(config_path: str) -> None:
    """
    Функция получения данных, предобработки и обучения модели
    :param config_path: расположение конфигурационного файла
    :return: None
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']

    train_config = config['train']

    train_data = load_data(train_path=preprocessing_config['train_path'])

    train_data = pipeline(data=train_data, flag_eval=False, **preprocessing_config)

    df_train, df_test = split_dataset(dataset=train_data, **preprocessing_config)

    study = optimal_params(data_train=df_train, data_test=df_test, **train_config)

    reg = train_model(
        data_train=df_train,
        data_test=df_test,
        study=study,
        target=preprocessing_config['target_column'],
        metric_path=train_config['metrics_path'],
    )

    joblib.dump(reg, os.path.join(train_config['model_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))