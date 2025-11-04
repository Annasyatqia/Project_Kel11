import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# ======================
# 🧠 LOAD MODEL & DATA
# ======================

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open('features.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_data():
    file_path = "DATA_WATER_SUPPLY_STATISTICS_clean.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning("⚠️ File dataset tidak ditemukan. Beberapa fitur analisis mungkin tidak berfungsi.")
        return None

# Load semua
model = load_model()
scaler = load_scaler()
features = load_features()
df = load_data()

# ======================
# 🎯 HEADER
# ======================
st.title("🌊 Prediksi Akses Air Bersih di Indonesia")

st.markdown("""
Aplikasi ini menggunakan **Machine Learning** untuk menganalisis dan memprediksi **akses air bersih di Indonesia**, 
sebagai bagian dari upaya mendukung **Tujuan Pembangunan Berkelanjutan (SDG 6: Air Bersih dan Sanitasi Layak)**.

💡 **Tujuan utama aplikasi ini:**
- Memantau perkembangan akses air bersih secara nasional.
- Memprediksi kebutuhan dan ketersediaan air bersih di tiap provinsi.
- Mengidentifikasi faktor-faktor utama yang mempengaruhi ketersediaan air bersih.
""")

# ======================
# 🧭 SIDEBAR MENU
# ======================
menu = st.sidebar.selectbox("Pilih Menu", ["Tren Nasional", "Prediksi Provinsi", "Analisis Faktor"])

# ======================
# 📈 TREND NASIONAL
# ======================
if menu == "Tren Nasional":
    st.header("📈 Tren Nasional Akses Air Bersih")
    st.markdown("""
    Pantau perkembangan rata-rata air bersih per tahun dan prediksi 5 tahun ke depan.
    Ini mendukung **SDG 6.1 (akses air minum layak dan terjangkau untuk semua)**.
    """)

    if df is not None:
        # Agregasi per tahun
        df_year = df.groupby('tahun')['air_bersih'].mean()
        df_year.index = pd.to_datetime(df_year.index, format='%Y')

        # Forecast model
        model_hw = ExponentialSmoothing(df_year, trend='add', seasonal=None, damped_trend=True)
        fit = model_hw.fit(optimized=True)
        forecast = fit.forecast(steps=5)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_year.index, y=df_year.values,
                                 mode='lines+markers', name='Aktual',
                                 line=dict(color='steelblue')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values,
                                 mode='lines+markers', name='Prediksi',
                                 line=dict(color='darkorange', dash='dash')))
        fig.update_layout(title="Tren dan Prediksi Jumlah Air Bersih (Rata-rata Nasional)",
                          xaxis_title="Tahun", yaxis_title="Jumlah Air Bersih")
        st.plotly_chart(fig)

        st.subheader("Hasil Prediksi 5 Tahun ke Depan")
        st.dataframe(forecast.round(2))
    else:
        st.info("📂 Unggah file `DATA_WATER_SUPPLY_STATISTICS_clean.csv` ke folder proyek
