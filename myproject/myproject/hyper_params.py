# 01 Import
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 02 Page Title
st.title('ハイパーパラメーター')

# 03 Load
df_params_t = pd.read_parquet("models/artifacts/LightGBM_params.parquet")

# 04 Show 
st.dataframe(df_params_t, use_container_width=True)