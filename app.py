# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px

st.set_page_config(layout="wide", page_title="Water Supply ML Dashboard")

@st.cache_data
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('features.json','r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

st.title("💧 Water Supply - Prediction & Dashboard")
st.markdown("Aplikasi untuk eksplorasi, prediksi `air_bersih` menggunakan Random Forest, dan forecasting tren.")

# Sidebar: upload dataset optional
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.write("Menggunakan dataset internal jika tersedia.")
    try:
        df = pd.read_csv('DATA_WATER_SUPPLY_STATISTICS.csv')
    except FileNotFoundError:
        st.error("Tidak menemukan DATA_WATER_SUPPLY_STATISTICS.csv pada repo. Upload file CSV atau tambahkan dataset ke repo.")
        st.stop()

# Basic show
if st.checkbox("Tampilkan 5 baris data"):
    st.dataframe(df.head())

# Preprocessing minimal (sama seperti train)
def preprocess_for_model(df):
    df = df.copy()
    df = df.rename(columns=lambda x: x.strip())
    # ensure features exist
    X = df.copy()
    # Numeric conversions like in training
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).str.replace(',', '.', regex=False).str.replace(r'[^0-9.\-]', '', regex=True)
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.select_dtypes(include=[np.number])
    # keep only features required by model (fill missing with 0)
    X = X.reindex(columns=features, fill_value=0)
    return X

st.sidebar.header("Prediction")
if st.sidebar.button("Predict on Uploaded / Repo data sample"):
    X = preprocess_for_model(df)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df_pred = df.copy()
    df_pred['pred_air_bersih'] = preds
    st.success("Prediksi selesai — menampilkan 10 baris pertama")
    st.dataframe(df_pred.head(10))
    # Show simple metrics if actual target present
    if 'air_bersih' in df.columns:
        y_true = pd.to_numeric(df['air_bersih'], errors='coerce')
        mask = ~y_true.isna()
        if mask.sum() > 0:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(y_true[mask], df_pred.loc[mask, 'pred_air_bersih'])
            rmse = np.sqrt(mean_squared_error(y_true[mask], df_pred.loc[mask, 'pred_air_bersih']))
            mae = mean_absolute_error(y_true[mask], df_pred.loc[mask, 'pred_air_bersih'])
            st.write(f"R2: {r2:.3f}  RMSE: {rmse:.2f}  MAE: {mae:.2f}")

# EDA view
st.header("Exploratory Data Analysis")
col1, col2 = st.columns([2,1])
with col1:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sel = st.selectbox("Pilih variabel numerik untuk histogram", numeric_cols)
    fig = plt.figure(figsize=(6,3))
    sns.histplot(df[sel].dropna(), kde=True)
    st.pyplot(fig)
with col2:
    st.write("Statistik ringkas")
    st.write(df.describe().T)

# Time series forecast (agregasi per tahun)
st.header("Forecast Tren (Holt-Winters)")
if 'tahun' in df.columns:
    # ensure tahun as year int or datetime
    try:
        ds = pd.to_datetime(df['tahun'], errors='coerce').dt.year
    except:
        ds = df['tahun']
    ts = df.copy()
    ts['tahun'] = ds
    ts_year = ts.groupby('tahun')['air_bersih'].mean().dropna()
    if len(ts_year) >= 3:
        st.write("Data tren per tahun")
        st.line_chart(ts_year)
        # fit model
        try:
            ts_year.index = pd.to_datetime(ts_year.index.astype(int).astype(str), format='%Y')
            model_hw = ExponentialSmoothing(ts_year, trend='add', seasonal=None, damped_trend=True)
            fit = model_hw.fit(optimized=True)
            fcast = fit.forecast(steps=5)
            st.write("Forecast 5 Tahun:")
            st.dataframe(fcast.round(3))
            # plot with plotly for nicer view
            fig = px.line()
            fig.add_scatter(x=ts_year.index, y=ts_year.values, mode='lines+markers', name='Actual')
            fig.add_scatter(x=fcast.index, y=fcast.values, mode='lines+markers', name='Forecast')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Forecast error: {e}")
    else:
        st.info("Data tahun kurang untuk forecasting (minimal 3 tahun diperlukan).")
else:
    st.info("Kolom 'tahun' tidak ditemukan di dataset.")

st.markdown("---")
st.write("Created from original Colab pipeline. Untuk retrain, jalankan `python train.py` lalu commit `model.pkl` & `scaler.pkl` ke repo.")
