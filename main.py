import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.copy_on_write = True

from src.features import create_features, build_advanced_features
from src.optimizer import optimize_lightgbm, optimize_catboost
from src.model import train_final_models, predict_ensemble, post_process_predictions
from src.utils import save_submission, prepare_data


def main():
    # Загрузка данных
    train = pd.read_csv('dataset/train_dataset.csv')
    test = pd.read_csv('dataset/valid_features.csv')
    target_col = 'Выработка. Результирующий расчет'
    
    # Feature Engineering
    print("Creating features...")
    train_feat = create_features(train)
    test_feat = create_features(test)
    train_median = train_feat.median(numeric_only=True)
    
    train_feat = build_advanced_features(train_feat, train_median)
    test_feat = build_advanced_features(test_feat, train_median)
    
    # Подготовка данных
    X, y, test_X = prepare_data(train_feat, test_feat, target_col)
    
    # Оптимизация гиперпараметров
    best_lgb = optimize_lightgbm(X, y, n_trials=30)
    best_cb = optimize_catboost(X, y, n_trials=20)
    
    # Обучение финальных моделей
    models = train_final_models(X, y, best_lgb, best_cb)
    
    # Предсказание
    test_preds = predict_ensemble(models, test_X, weights=(0.45, 0.45, 0.1))
    
    # Пост-обработка
    test_preds = post_process_predictions(test_preds, test_X)
    
    # Сохранение
    save_submission(test_preds, 'submission.csv')
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()