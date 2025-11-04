# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ======================
# ⚙️ KONFIGURASI DASAR
# ======================
st.set_page_config(
    page_title="💧 Dashboard SDG 6 – Air Bersih Indonesia",
    layout="wide",
    page_icon="💧"
)

# ======================
# 📦 LOAD DATA DAN MODEL
# ======================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_features():
    with open("features.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_year_data():
    return pd.read_csv("df_year.csv")

model = load_model()
scaler = load_scaler()
features = load_features()
df_year = load_year_data()

# ======================
# 🎯 HEADER
# ======================
st.title("💧 Prediksi & Analisis Akses Air Bersih di Indonesia")
st.markdown("""
Aplikasi ini dikembangkan untuk mendukung **SDG 6: Air Bersih dan Sanitasi Layak**  
melalui analisis **tren, prediksi provinsi, dan faktor penentu utama akses air bersih**.  
""")

st.divider()

# ======================
# 🔹 PILIH HALAMAN
# ======================
menu = st.sidebar.radio(
    "Navigasi Halaman",
    ["Tren Nasional", "Prediksi Provinsi", "Analisis Faktor"]
)

# ==========================================================
# 1️⃣ TREN NASIONAL — Pemantauan SDG 6.1 & 6.4
# ==========================================================
if menu == "Tren Nasional":
    st.header("📈 Tren Nasional Akses Air Bersih")
    st.markdown("""
    Menampilkan perkembangan historis dan proyeksi **5 tahun ke depan** berdasarkan data `df_year.csv`.  
    Analisis ini mendukung pemantauan **SDG 6.1 (akses air bersih)** dan **SDG 6.4 (efisiensi penggunaan air)**.
    """)

    provinsi_list = sorted(df_year["Provinsi"].unique())
    prov_pilih = st.selectbox("Pilih Provinsi", ["Semua Provinsi"] + provinsi_list)

    # Filter data
    if prov_pilih == "Semua Provinsi":
        df_plot = df_year.groupby("Tahun")["Jumlah_Air_Bersih"].sum().reset_index()
        judul = "Tren Nasional"
    else:
        df_plot = df_year[df_year["Provinsi"] == prov_pilih]
        judul = f"Tren Provinsi {prov_pilih}"

    # Plot tren historis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_plot["Tahun"], df_plot["Jumlah_Air_Bersih"], marker="o", linewidth=2)
    ax.set_title(judul)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Air Bersih (unit/liter)")
    st.pyplot(fig)

    # Prediksi sederhana 5 tahun ke depan (linear)
    df_plot = df_plot.sort_values("Tahun")
    x = np.arange(len(df_plot))
    y = df_plot["Jumlah_Air_Bersih"].values
    coef = np.polyfit(x, y, 1)
    pred_line = np.poly1d(coef)

    tahun_lanjut = np.arange(len(df_plot), len(df_plot) + 5)
    tahun_pred = np.arange(df_plot["Tahun"].iloc[-1] + 1, df_plot["Tahun"].iloc[-1] + 6)
    pred_y = pred_line(tahun_lanjut)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df_plot["Tahun"], y, marker="o", label="Data Aktual")
    ax2.plot(tahun_pred, pred_y, "--o", label="Prediksi 5 Tahun ke Depan")
    ax2.legend()
    ax2.set_title(f"Proyeksi Air Bersih – {judul}")
    st.pyplot(fig2)

    st.info("💡 Prediksi ini bersifat sederhana dan dapat dikembangkan menggunakan model time-series seperti ARIMA atau Prophet untuk hasil lebih akurat.")

# ==========================================================
# 2️⃣ PREDIKSI PROVINSI — Simulasi Kebijakan Daerah
# ==========================================================
elif menu == "Prediksi Provinsi":
    st.header("🔮 Prediksi Akses Air Bersih per Provinsi")
    st.markdown("""
    Model Machine Learning digunakan untuk memperkirakan **persentase akses air bersih**
    berdasarkan indikator sosial–ekonomi dan operasional perusahaan air bersih.
    """)

    # Pilihan provinsi & tahun
    provinsi = st.selectbox("Pilih Provinsi", sorted(df_year["Provinsi"].unique()))
    tahun = st.number_input("Masukkan Tahun", min_value=2000, max_value=2100, value=2025)

    st.markdown("### 🧾 Input Indikator")
    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        col = cols[i % 3]
        user_input[feat] = col.number_input(feat, value=0.0)

    input_df = pd.DataFrame([user_input])
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Jalankan Prediksi"):
        try:
            X_scaled = scaler.transform(input_df[features])
            pred = model.predict(X_scaled)[0]

            st.success(f"💧 Perkiraan Akses Air Bersih: {pred:.2f}%")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(["Prediksi Akses Air Bersih"], [pred], color="#56CCF2")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Persentase (%)")
            st.pyplot(fig)

            if pred < 60:
                st.error("⚠️ Akses air bersih rendah – perlu peningkatan infrastruktur dan efisiensi produksi.")
            elif pred < 80:
                st.warning("🟡 Akses cukup – masih perlu perbaikan di daerah pedesaan dan kelompok sosial.")
            else:
                st.success("✅ Akses tinggi – fokus pada keberlanjutan dan pemeliharaan sumber air.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# ==========================================================
# 3️⃣ ANALISIS FAKTOR — Menentukan Intervensi Prioritas
# ==========================================================
elif menu == "Analisis Faktor":
    st.header("🧠 Analisis Faktor Penting (Feature Importance)")
    st.markdown("""
    Menunjukkan **variabel yang paling memengaruhi akses air bersih**,  
    untuk membantu pemerintah menentukan **prioritas intervensi kebijakan**.
    """)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance["Feature"], importance["Importance"], color="#4DB6AC")
        ax.set_title("Feature Importance – Faktor Penentu Akses Air Bersih")
        st.pyplot(fig)

        top_factors = importance.sort_values(by="Importance", ascending=False).head(5)
        st.markdown("### 🔍 Lima Faktor Teratas:")
        for i, row in top_factors.iterrows():
            st.markdown(f"- **{row['Feature']}** → kontribusi {row['Importance']:.3f}")
    else:
        st.warning("Model tidak memiliki atribut feature_importances_. Gunakan model berbasis pohon (misal RandomForest atau XGBoost) agar fitur ini aktif.")

st.divider()

# ======================
# 🌍 PENUTUP
# ======================
st.caption("""
📊 **SDG 6.1 & 6.4 – Air Bersih dan Efisiensi Penggunaan Air**  
Aplikasi ini mendukung pengambilan keputusan berbasis data dalam perencanaan infrastruktur air bersih.
""")
