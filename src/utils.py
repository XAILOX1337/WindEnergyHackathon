import numpy as np
import pandas as pd


def save_submission(preds, name='submission.csv'):
    """Сохранение submission файла."""
    preds = np.asarray(preds).ravel()
    if len(preds) != 2126:
        print(f"WARNING: Expected 2126 rows, got {len(preds)}. Check indices.")
    pd.DataFrame(preds).to_csv(name, index=False, header=False)
    print(f"Submission saved: {name}")


def prepare_data(train_feat, test_feat, target_col):
    """Подготовка X, y для обучения."""
    drop_cols = ['datetime', 'METEOFORECASTHOUR_OPENM_Datetime', target_col]
    
    X = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
    y = train_feat[target_col]
    X = X.select_dtypes(include='number')
    
    test_X = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns], errors='ignore')
    test_X = test_X.select_dtypes(include='number')
    
    # Синхронизация колонок
    common_cols = X.columns.intersection(test_X.columns)
    X = X[common_cols]
    test_X = test_X[common_cols]
    
    return X, y, test_X