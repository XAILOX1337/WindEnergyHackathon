import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error


def objective_lgb(trial, X, y):
    """Optuna objective для LightGBM."""
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for tr_idx, val_idx in tscv.split(X):
        model = lgb.LGBMRegressor(**param, n_estimators=2000)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = np.clip(model.predict(X.iloc[val_idx]), 0, 90.09)
        scores.append(root_mean_squared_error(y.iloc[val_idx], preds))
    
    return np.mean(scores)


def objective_cb(trial, X, y):
    """Optuna objective для CatBoost."""
    param = {
        "iterations": 1500,
        "random_seed": 42,
        "thread_count": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 1.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for tr_idx, val_idx in tscv.split(X):
        model = CatBoostRegressor(**param, verbose=0, early_stopping_rounds=50)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx])
        )
        preds = np.clip(model.predict(X.iloc[val_idx]), 0, 90.09)
        scores.append(root_mean_squared_error(y.iloc[val_idx], preds))
    
    return np.mean(scores)


def optimize_lightgbm(X, y, n_trials=30):
    """Запуск оптимизации LightGBM."""
    print("Optimizing LightGBM...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_lgb(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    })
    return best_params


def optimize_catboost(X, y, n_trials=20):
    """Запуск оптимизации CatBoost."""
    print("Optimizing CatBoost...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_cb(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    best_params.update({
        "random_seed": 42,
        "thread_count": -1,
        "verbose": 0,
        "early_stopping_rounds": 100
    })
    return best_params