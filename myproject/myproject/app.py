# 01 Import
import streamlit as st
from streamlit_option_menu import option_menu
import os

# 02 Page Layout
st.set_page_config(page_title='ホーム', page_icon='🏠', layout='wide')


# 03 Side Bar
with st.sidebar:
    selected = option_menu('Menu', ['ホーム',
                                    'スポット市場直近データ', 
                                    'スポット市場価格予測', 
                                    '予測精度検証'], 
        icons=['bi-house-door-fill', 'bi-database-fill-check', 'bi-graph-up', 'bi-file-earmark-check-fill'], 
        menu_icon='cast', default_index=0)


# 04 Pagenation
# Get the directry where `app.py` exists
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Page setting
if selected == 'ホーム':
    exec(open(os.path.join(APP_DIR, 'home.py'), encoding='utf-8').read())
    
elif selected == 'スポット市場直近データ':
    exec(open(os.path.join(APP_DIR, 'spot_latest.py'), encoding='utf-8').read())
    
elif selected == 'スポット市場価格予測':
    exec(open(os.path.join(APP_DIR, 'spot_forecast.py'), encoding='utf-8').read())
    
elif selected == '予測精度検証':
    exec(open(os.path.join(APP_DIR, 'forecast_eval.py'), encoding='utf-8').read())
    