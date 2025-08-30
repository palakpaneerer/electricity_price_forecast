# 01 Import
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 02 Page Title
st.title('ハイパーパラメーター')

# 03 Load
APP_DIR = os.path.dirname(os.path.abspath(__file__))

df_params_t_path = os.path.join(APP_DIR, 'models', 'artifacts', 'LightGBM_params.parquet')
df_params_t = pd.read_parquet(df_params_t_path, engine='pyarrow')

# 04 Show 
st.dataframe(df_params_t, use_container_width=True)