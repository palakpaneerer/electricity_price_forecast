# 01 Import
import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import timedelta

import os

from io import BytesIO
import plotly.express as px

import xgboost as xgb


# 02 Page Title
st.title('スポット市場価格予測')
st.text('''
        ・XGBoostで予測を行っています。
        ・モデルの精度は「予測精度検証」ページでご確認ください。
        ''')


# 03 Extract Data
dfs = {}

years = [2024, 2025]
for year in years:
    url = "https://www.jepx.jp/_download.php"
    params = {'dir': 'spot_summary', 'file': f'spot_summary_{year}.csv'}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'https://www.jepx.jp/electricpower/market-data/spot/'
    }

    response = requests.post(url, data=params, headers=headers)
    response.raise_for_status()

    csv_bytes = response.content
    df = pd.read_csv(BytesIO(csv_bytes), encoding='shift_jis')

    dfs[year] = df

# Merge
df_all = pd.concat(dfs.values(), ignore_index=True)


# 04 Extend the df dates
# Check latest date
last_date = pd.to_datetime(df_all['受渡日'].max())
last_code = df_all['時刻コード'].max()

# Create future times by 30 mins for7 
future_steps = 48 * 7  # 336
future_times = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=future_steps // 48,
    freq='D'
)

# Create future rows by using combinations between date and timecode
future_rows = []
for day in future_times:
    for code in range(1, 49):  # 1〜48 timecode
        future_rows.append({
            '受渡日': day,
            '時刻コード': code
        })

# Make the future rows DataFrame
df_future = pd.DataFrame(future_rows)

# Add all same coluns from df_all
for col in df_all.columns:
    if col not in df_future.columns:
        df_future[col] = None

df_future = df_future[df_all.columns]

# Merge the dataframes
df_extended = pd.concat([df_all, df_future], ignore_index=True)


# 05 Data Cleaning
df_extended['受渡日'] = pd.to_datetime(df_extended['受渡日'], errors='coerce')

df_extended['受渡日'] = pd.to_datetime(df_extended['受渡日'])
df_extended['時刻'] = (df_extended['時刻コード'] - 1) * 30
df_extended['時刻'] = pd.to_timedelta(df_extended['時刻'], unit='m')
df_extended['受渡日時'] = df_extended['受渡日'] + df_extended['時刻']


# 06 Feature Engineering
# Date features
df_extended['year'] = df_extended['受渡日'].dt.year
df_extended['month'] = df_extended['受渡日'].dt.month
df_extended['day_of_week'] = df_extended['受渡日'].dt.day_of_week
df_extended['day_of_year'] = df_extended['受渡日'].dt.dayofyear

target_cols = [
    'エリアプライス北海道(円/kWh)',
    'エリアプライス東北(円/kWh)',
    'エリアプライス東京(円/kWh)',
    'エリアプライス中部(円/kWh)',
    'エリアプライス北陸(円/kWh)',
    'エリアプライス関西(円/kWh)',
    'エリアプライス中国(円/kWh)',
    'エリアプライス四国(円/kWh)',
    'エリアプライス九州(円/kWh)',
]
for target in target_cols:
    df_extended[f'{target}_lag336'] = df_extended[target].shift(336)
    df_extended[f'{target}_r336'] = df_extended[target].rolling(336).mean()
    
    
# Cyclical encoding for periodic features
df_extended['month_sin'] = np.sin(2*np.pi*(df_extended['month']-1)/12)
df_extended['month_cos'] = np.cos(2*np.pi*(df_extended['month']-1)/12)

df_extended['dow_sin'] = np.sin(2*np.pi*df_extended['day_of_week']/7)
df_extended['dow_cos'] = np.cos(2*np.pi*df_extended['day_of_week']/7)

df_extended['days_in_year'] = np.where(df_extended['受渡日'].dt.is_leap_year, 366, 365)
df_extended['frac_of_year'] = (df_extended['day_of_year'] - 1) / df_extended['days_in_year']
df_extended['doy_sin'] = np.sin(2 * np.pi * df_extended['frac_of_year'])
df_extended['doy_cos'] = np.cos(2 * np.pi * df_extended['frac_of_year'])

df_extended['tod_sin'] = np.sin(2*np.pi*df_extended['時刻コード']/48)
df_extended['tod_cos'] = np.cos(2*np.pi*df_extended['時刻コード']/48)


# 07 Keep only the latest 2 weeks (48 slots/day × 14 days = 672 rows)
df_extended = df_extended.tail(48 * 14).reset_index(drop=True)


# 08 User Can Select Areas
area_map = {
    '北海道': 'エリアプライス北海道(円/kWh)',
    '東北':   'エリアプライス東北(円/kWh)',
    '東京':   'エリアプライス東京(円/kWh)',
    '中部':   'エリアプライス中部(円/kWh)',
    '北陸':   'エリアプライス北陸(円/kWh)',
    '関西':   'エリアプライス関西(円/kWh)',
    '中国':   'エリアプライス中国(円/kWh)',
    '四国':   'エリアプライス四国(円/kWh)',
    '九州':   'エリアプライス九州(円/kWh)',
}

options = list(area_map.keys())

selected = st.multiselect(
    'エリアを選択してください',
    options,
    default='東京'
)

selected_cols = [area_map[opt] for opt in selected]


df_all_pred = pd.DataFrame({"受渡日時": df_extended["受渡日時"]})
if not selected:
    st.warning("エリアを1つ以上選択してください。")
else:
    plot_cols = []
    for area in selected:
        # A. Prediction
        # Access the model
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(APP_DIR, f"models/models_XGBoost/xgboost_{area}.json")
        model = xgb.Booster()
        model.load_model(model_path)


        # B Use the same features as the trained model
        target_col = f"エリアプライス{area}(円/kWh)"
        feature_cols = [
            'year', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'doy_sin', 'doy_cos', 'tod_sin', 'tod_cos',
            f'{target_col}_lag336', f'{target_col}_r336'
        ]

        X_input = df_extended[feature_cols].copy()
        dtest = xgb.DMatrix(X_input)


        # C Predict
        y_pred = model.predict(dtest)

        
        # D Save the prediction and actual price
        pred_col_name = f'{area}_予測値'
        actual_col_name = f'{area}_実測値'

        df_all_pred[pred_col_name] = y_pred
        df_all_pred[actual_col_name] = df_extended[target_col].values

        
        # E Add cols that I want to show in the line graph
        plot_cols.append(actual_col_name)
        plot_cols.append(pred_col_name)
    
    
    
# Show only selected areas for graph
fig = px.line(
    df_all_pred,
    x='受渡日時',
    y=plot_cols,
    labels={'受渡日時': '受渡日時', 'value': '価格(円/kWh)', 'variable': 'エリア'}
)
st.plotly_chart(fig, use_container_width=True)