# 01 Import
import streamlit as st
import requests
import pandas as pd
from io import BytesIO

import plotly.express as px


# 02 Page Title
st.title('スポット市場価格予測')


# 03 LightGBM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import holidays
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
SEED = 86

# 04 Extract Data
url = 'https://www.jepx.jp/_download.php'
params = {'dir': 'spot_summary', 'file': 'spot_summary_2025.csv'}
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Referer': 'https://www.jepx.jp/electricpower/market-data/spot/'
}
response = requests.post(url, data=params, headers=headers)
csv_bytes = response.content
df = pd.read_csv(BytesIO(csv_bytes), encoding='shift_jis')

# Data Cleaning
df['受渡日'] = pd.to_datetime(df['受渡日'])
df['時刻'] = (df['時刻コード'] - 1) * 30
df['時刻'] = pd.to_timedelta(df['時刻'], unit='m')
df['受渡日時'] = df['受渡日'] + df['時刻']

# Feature Engineering
df['year'] = df['受渡日'].dt.year
df['quarter'] = df['受渡日'].dt.quarter
df['month'] = df['受渡日'].dt.month
df['day_of_week'] = df['受渡日'].dt.day_of_week
df['week_number'] = df['受渡日'].dt.isocalendar().week
df['day_of_year'] = df['受渡日'].dt.dayofyear

# Training
target = 'エリアプライス東京(円/kWh)'
select_column = [
    'year', 'quarter', 'month', '時刻コード', 'day_of_week', 'week_number', 'day_of_year'
]

df_train = df[:-48*7]
df_val = df_train[-48*7:]
df_test = df[-48*7:]

X_train = df_train[select_column]
y_train = df_train[target]

X_val = df_val[select_column]
y_val = df_val[target]

X_test = df_test[select_column]
y_test = df_test[target]

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators':20000,
    'max_depth': -1,
    'learning_rate': 0.01,
    'num_leaves': 480,
    'feature_fraction': 0.25,
    'random_state': SEED,
    'verbosity': -1,
}

model = lgb.LGBMRegressor(**params)

model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    eval_metric = 'rmse', 
    callbacks = [
        lgb.early_stopping(50000),
        lgb.log_evaluation(1000)
    ]
)

# Test - 再学習していない
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

st.write(f'Test RMSE: {rmse:.4f}')
st.write(f'Test MAE: {mae:.4f}')


# User Can Select Areas
# area_map = {
#     '北海道': 'エリアプライス北海道(円/kWh)',
#     '東北':   'エリアプライス東北(円/kWh)',
#     '東京':   'エリアプライス東京(円/kWh)',
#     '中部':   'エリアプライス中部(円/kWh)',
#     '北陸':   'エリアプライス北陸(円/kWh)',
#     '関西':   'エリアプライス関西(円/kWh)',
#     '中国':   'エリアプライス中国(円/kWh)',
#     '四国':   'エリアプライス四国(円/kWh)',
#     '九州':   'エリアプライス九州(円/kWh)',
# }

# options = list(area_map.keys())


# selected = st.multiselect(
#     'エリアを選択してください',
#     options,
#     default=options
# )

# selected_cols = [area_map[opt] for opt in selected]
selected_cols='エリアプライス東京(円/kWh)'
if not selected_cols:
    st.warning("エリアを1つ以上選択してください。")
else:
    # Show only selected areas
    fig = px.line(
        df,
        x='受渡日時',
        y=selected_cols,
        labels={'受渡日時': '受渡日時', 'value': '価格(円/kWh)', 'variable': 'エリア'},
        title="選択されたエリアのスポット価格"
    )
    st.plotly_chart(fig, use_container_width=True)





plot_df = df_test[['受渡日時', target]].reset_index(drop=True).copy()
plot_df['予測値'] = y_pred

st.write(plot_df)


# ロング形式に変換（系列: 実測 or 予測, 価格: 値）
long_df = plot_df.melt(
    id_vars='受渡日時',
    value_vars=[target, '予測値'],
    var_name='系列',
    value_name='価格(円/kWh)'
)

# 可視化
fig_pred = px.line(
    long_df,
    x='受渡日時',
    y='価格(円/kWh)',
    color='系列',
    title='実測 vs 予測（テスト期間）'
)
st.plotly_chart(fig_pred, use_container_width=True)
# st.table(df.head(10))