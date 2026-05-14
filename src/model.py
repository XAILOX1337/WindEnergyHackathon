import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def train_final_models(X, y, best_lgb, best_cb):
    """Обучение финальных моделей на полном датасете."""
    print("Training final models on full dataset...")
    
    final_lgb = lgb.LGBMRegressor(**best_lgb).fit(X, y)
    final_cb = CatBoostRegressor(**best_cb).fit(X, y, logging_level='Silent')
    final_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ).fit(X, y)
    
    return final_lgb, final_cb, final_xgb


def predict_ensemble(models, test_X, weights=(0.45, 0.45, 0.1)):
    """Ансамблевое предсказание с весами."""
    final_lgb, final_cb, final_xgb = models
    
    p_lgb = final_lgb.predict(test_X)
    p_cb = final_cb.predict(test_X)
    p_xgb = final_xgb.predict(test_X.values)
    
    test_preds = (p_lgb * weights[0]) + (p_cb * weights[1]) + (p_xgb * weights[2])
    return test_preds


def post_process_predictions(preds, test_X):
    """Пост-обработка: физические ограничения."""
    test_preds = np.asarray(preds).ravel()
    
    if 'wind_speed_80m' in test_X.columns:
        # Cut-in speed
        test_preds[test_X['wind_speed_80m'].values < 2.5] = 0
        # Cut-out speed
        test_preds[test_X['wind_speed_80m'].values > 25] = 0
    
    # Физический лимит мощности
    test_preds = np.clip(test_preds, 0, 90.09)
    return test_preds