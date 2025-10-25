# 01 Import
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 02 Page Title
st.title('予測精度検証')
st.text('''
        ・学習データ: 2022年1月1日 - 2025年10月17日
        ・検証データ: 2025年10月17日 - 2025年10月24日
        ・参考: 論文では月間予測において九州地方でMAE 1.936が報告されています。
        ・https://masahiromae.github.io/pdf/journal/202406_Energies_WangMaeMatsuhashi.pdf
        ''')

# 03 Load
APP_DIR = os.path.dirname(os.path.abspath(__file__))

mae_parquet_path = os.path.join(APP_DIR, 'models', 'artifacts', 'df_mae_results.parquet')
df_mae_results = pd.read_parquet(mae_parquet_path, engine='pyarrow')

rmse_parquet_path = os.path.join(APP_DIR, 'models', 'artifacts', 'df_rmse_results.parquet')
df_rmse_results = pd.read_parquet(rmse_parquet_path, engine='pyarrow')

# 04 Def Heatmap 
def make_heatmap(df: pd.DataFrame, title: str):

    mat = df.set_index("Region")

    fig = px.imshow(
        mat,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn_r"  # 低エラー=緑 / 高エラー=赤
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Region",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

# 05 Show heatmaps
st.plotly_chart(make_heatmap(df_mae_results, "MAE Heatmap"), use_container_width=True)
st.plotly_chart(make_heatmap(df_rmse_results, "RMSE Heatmap"), use_container_width=True)