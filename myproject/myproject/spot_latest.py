# 参考: https://qiita.com/InvestorX/items/f1649d046a8405bdca8e#:~:text=%3Cselect%20class%3D%22modal,2018%E5%B9%B4%E5%BA%A6%3C%2Foption

# 01 Import
import streamlit as st
import requests
import pandas as pd
from io import BytesIO

import plotly.express as px


# 02 Page Title
st.title('スポット市場直近7日間データ')


# 03 Extract Data
url = 'https://www.jepx.jp/_download.php'
params = {'dir': 'spot_summary', 'file': 'spot_summary_2025.csv'}
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Referer': 'https://www.jepx.jp/electricpower/market-data/spot/'
}

response = requests.post(url, data=params, headers=headers)
csv_bytes = response.content
df = pd.read_csv(BytesIO(csv_bytes), encoding='shift_jis')


# 04 Data Cleaning
df['受渡日'] = pd.to_datetime(df['受渡日'])
df['時刻'] = (df['時刻コード'] - 1) * 30
df['時刻'] = pd.to_timedelta(df['時刻'], unit='m')
df['受渡日時'] = df['受渡日'] + df['時刻']


# 05 Extract Only 7 Days Data
end_date = df['受渡日'].max()
start_date = end_date - pd.Timedelta(days=6)
mask = (df['受渡日'] >= start_date) & (df['受渡日'] <= end_date)
df7 = df.loc[mask].copy()


# 06 User Can Select Areas
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
    default=options
)

selected_cols = [area_map[opt] for opt in selected]

if not selected_cols:
    st.warning("エリアを1つ以上選択してください。")
else:
    # Show only selected areas
    fig = px.line(
        df7,
        x='受渡日時',
        y=selected_cols,
        labels={'受渡日時': '受渡日時', 'value': '価格(円/kWh)', 'variable': 'エリア'},
        title="選択されたエリアのスポット価格"
    )
    st.plotly_chart(fig, use_container_width=True)