"""
Программа: Обучение модели
Версия: 1.0
"""

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna import Study
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ..data.split_dataset import train_test_data
from .metrics import save_metrics


def best_params(trial, X, y, N_FOLDS, RANDOM_STATE):
    """
    Функция подбора оптимальных гиперпараметров
    :param trial: кол-во trials
    :param X: данные с признакамит без целевой переменной
    :param y: целевая переменная
    :param N_FOLDS: кол-во фолдов
    :param RANDOM_STATE: random_state
    :return: среднее значение метрик по фолдам
    """
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [500]),
        "random_state": trial.suggest_categorical("random_state", [RANDOM_STATE]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 15, step=1),
        "reg_alpha": trial.suggest_int("lambda_l1", 0, 100),
        "reg_lambda": trial.suggest_int("lambda_l2", 0, 100),
        "min_split_gain": trial.suggest_int("min_gain_to_split", 0, 10),
        "subsample": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "subsample_freq": trial.suggest_categorical("bagging_freq", [1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0)
    }

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_predicts = np.empty(N_FOLDS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")

        model = LGBMRegressor(**params)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric="mae",
                  callbacks=[pruning_callback]
                  )

        preds = model.predict(X_test)
        cv_predicts[idx] = mean_absolute_error(y_test, preds)

    return np.mean(cv_predicts)


def optimal_params(data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs) -> Study:
    """
    Пайплайн тренировки модели с подбором
    оптимальных гиперпараметров
    :param data_train: train данные
    :param data_test: test данные
    :return: [LGBMRegressor tuning, Study]
    """
    x_train, x_test, y_train, y_test = train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: best_params(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        study: Study,
        target: str,
        metric_path: str,
) -> LGBMRegressor:
    """
    Обучение модели на оптимальных гиперпараметрах
    :param data_train: train данные
    :param data_test: test данные
    :param study: study optuna
    :param target: целевая переменная
    :param metric_path: месторасположение для сохранения метрик
    :return: модель с оптимальными гиперпараметрами
    """
    x_train, x_test, y_train, y_test = train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    model_tune = LGBMRegressor(**study.best_params)
    model_tune.fit(x_train, y_train)

    save_metrics(data_x=x_test, data_y=y_test, model=model_tune, metric_path=metric_path)
    return model_tune
