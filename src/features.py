import pandas as pd
import numpy as np


def wind_power_curve(v, v_cut_in=2.5, v_rated=12.5, v_cut_out=25.0, p_rated=90.09):
    """Расчет теоретической мощности ветрогенератора."""
    v = np.asarray(v)
    p = np.zeros_like(v)
    mask = (v >= v_cut_in) & (v < v_rated)
    p[mask] = p_rated * ((v[mask] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    p[(v >= v_rated) & (v < v_cut_out)] = p_rated
    return p


def create_features(df):
    """Базовый feature engineering."""
    df = df.copy()
    df['original_index'] = np.arange(len(df))
    df['datetime'] = pd.to_datetime(df['METEOFORECASTHOUR_OPENM_Datetime'])
    df = df.sort_values('datetime')
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['available_ratio'] = (26 - df['Кол-во_ВЭУ_в_ремонте']) / 26
    df['air_density'] = df['pressure_msl'] / (df['temperature_80m'] + 273.15)
    df['theoretical_power'] = wind_power_curve(df['wind_speed_80m'].values) * df['available_ratio']

    for w in [3, 6]:
        df[f'wind_speed_roll_mean_{w}'] = df['wind_speed_80m'].rolling(window=w).mean()
        df[f'wind_speed_roll_std_{w}'] = df['wind_speed_80m'].rolling(window=w).std()
        
    df = df.ffill().fillna(0)
    df = df.sort_values('original_index').drop(columns=['original_index'])
    return df


def build_advanced_features(df, train_median=None):
    """Продвинутые признаки: лаги, векторы ветра, градиенты."""
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