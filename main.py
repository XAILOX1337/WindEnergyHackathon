import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.copy_on_write = True

def create_features(df):
    df = df.copy()
    df['original_index'] = np.arange(len(df))
    df['datetime'] = pd.to_datetime(df['METEOFORECASTHOUR_OPENM_Datetime'])
    df = df.sort_values('datetime')
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['wind_speed_cube'] = df['wind_speed_80m'] ** 3
    df['available_ratio'] = (26 - df['Кол-во_ВЭУ_в_ремонте']) / 26
    df['air_density'] = df['pressure_msl'] / (df['temperature_80m'] + 273.15)
    
    # 1. Физическая мощность (энергия ветра)
    df['theoretical_power'] = df['air_density'] * df['wind_speed_cube'] * df['available_ratio']
    
    # 2. Скользящие показатели (Rolling) - тренды и волатильность
    for w in [3, 6]:
        df[f'wind_speed_roll_mean_{w}'] = df['wind_speed_80m'].rolling(window=w).mean()
        df[f'wind_speed_roll_std_{w}'] = df['wind_speed_80m'].rolling(window=w).std()
        
    # Заполнение NaN после окон
    df = df.ffill().fillna(0)
    df = df.sort_values('original_index').drop(columns=['original_index'])
    return df

def build_advanced_features(df, train_median=None):
    df = df.copy()
    if df['wind_direction_80m'].max() > 1.1:
        df['wind_dir_norm'] = df['wind_direction_80m'] / 360.0
    else:
        df['wind_dir_norm'] = df['wind_direction_80m']

    df['wind_u'] = df['wind_speed_80m'] * np.cos(2 * np.pi * df['wind_dir_norm'])
    df['wind_v'] = df['wind_speed_80m'] * np.sin(2 * np.pi * df['wind_dir_norm'])
    df.drop(columns=['wind_dir_norm'], inplace=True, errors='ignore')
    df['temp_gradient'] = df['temperature_120m'] - df['temperature_80m']

    for lag in [1, 2, 3, 24]:
        df[f'wind_speed_lag_{lag}'] = df['wind_speed_80m'].shift(lag)
        df[f'temp_80m_lag_{lag}'] = df['temperature_80m'].shift(lag)

    df = df.ffill()
    fill_values = train_median if train_median is not None else df.median(numeric_only=True)
    df = df.fillna(fill_values)
    return df

def save_submission(preds, name='submission.csv'):
    preds = np.asarray(preds).ravel()
    if len(preds) != 2126:
        print(f"WARNING: Expected 2126 rows, got {len(preds)}. Check indices.")
    pd.DataFrame(preds).to_csv(name, index=False, header=False)
    print(f"Submission saved: {name}")

# ==========================================
# ОСНОВНОЙ ПЛАЙПЛАЙН
# ==========================================
train = pd.read_csv('dataset/train_dataset.csv')
test = pd.read_csv('dataset/valid_features.csv')
target_col = 'Выработка. Результирующий расчет'

# 1. Feature Engineering
train_feat = create_features(train)
test_feat = create_features(test)
train_median = train_feat.median(numeric_only=True)
train_feat = build_advanced_features(train_feat, train_median)
test_feat = build_advanced_features(test_feat, train_median)

# 2. Подготовка X, y (без TE, чтобы избежать размерностных конфликтов и leakage)
drop_cols = ['datetime', 'METEOFORECASTHOUR_OPENM_Datetime', target_col]
X = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
y = train_feat[target_col]
X = X.select_dtypes(include='number')

test_X = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns], errors='ignore')
test_X = test_X.select_dtypes(include='number')
common_cols = X.columns.intersection(test_X.columns)
X = X[common_cols]
test_X = test_X[common_cols]

# 3. Optuna: LightGBM
def objective_lgb(trial):
    param = {
        "objective": "regression", "metric": "rmse", "verbosity": -1, "boosting_type": "gbdt",
        "random_state": 42, "n_jobs": -1,
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
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], 
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = np.clip(model.predict(X.iloc[val_idx]), 0, 90.09)
        scores.append(root_mean_squared_error(y.iloc[val_idx], preds))
    return np.mean(scores)

print("Optimizing LightGBM...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)
best_lgb = study_lgb.best_params.copy()
best_lgb.update({"n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbosity": -1})

# 4. Optuna: CatBoost (с фокусом на регуляризацию)
def objective_cb(trial):
    param = {
        "iterations": 1500, "random_seed": 42, "thread_count": -1,
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
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        preds = np.clip(model.predict(X.iloc[val_idx]), 0, 90.09)
        scores.append(root_mean_squared_error(y.iloc[val_idx], preds))
    return np.mean(scores)

print("Optimizing CatBoost...")
study_cb = optuna.create_study(direction="minimize")
study_cb.optimize(objective_cb, n_trials=20, show_progress_bar=True)
best_cb = study_cb.best_params.copy()
best_cb.update({"random_seed": 42, "thread_count": -1, "verbose": 0, "early_stopping_rounds": 100})

# 5. Финальное обучение всех базовых моделей
print("Training final models on full dataset...")
final_lgb = lgb.LGBMRegressor(**best_lgb).fit(X, y)
final_cb = CatBoostRegressor(**best_cb).fit(X, y, logging_level='Silent')
final_xgb = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=8, 
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, 
                            random_state=42, n_jobs=-1, verbosity=0).fit(X, y)

# 6. Предсказание и взвешенный ансамбль
p_lgb = final_lgb.predict(test_X)
p_cb = final_cb.predict(test_X)
p_xgb = final_xgb.predict(test_X.values)

# Взвешивание на основе CV-производительности моделей
test_preds = (p_lgb * 0.45) + (p_cb * 0.45) + (p_xgb * 0.1)

# 7. Пост-обработка (Hard Rules)
test_preds = np.asarray(test_preds).ravel()
if 'wind_speed_80m' in test_X.columns:
    # Cut-in speed: турбина не вырабатывает энергию при штиле
    test_preds[test_X['wind_speed_80m'].values < 2.5] = 0
    # Cut-out speed: защита от шторма
    test_preds[test_X['wind_speed_80m'].values > 25] = 0

# Физический лимит мощности
test_preds = np.clip(test_preds, 0, 90.09)
save_submission(test_preds, 'submission.csv')
print("Pipeline completed successfully.")