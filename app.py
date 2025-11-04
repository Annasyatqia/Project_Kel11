import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# 1. SETUP PAGE
# ============================================================
st.set_page_config(
    page_title="Analisis Akses Air Bersih Jawa Barat",
    layout="wide",
    page_icon="💧"
)

st.title("💧 Analisis Akses Air Bersih di Jawa Barat")
st.markdown("Aplikasi ini digunakan untuk menganalisis dan memprediksi akses air bersih di Jawa Barat berdasarkan data statistik yang tersedia.")

# ============================================================
# 2. LOAD DATASET MENTAH
# ============================================================
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

data_path = "DATA_WATER_SUPPLY_STATISTICS.csv"
if os.path.exists(data_path):
    df_raw = load_data(data_path)
    st.success("✅ Dataset berhasil dimuat!")
else:
    st.warning("⚠️ File `DATA_WATER_SUPPLY_STATISTICS.csv` belum ada di folder proyek.")
    st.stop()

# ============================================================
# 3. CLEANING OTOMATIS DI APP
# ============================================================
st.subheader("🧹 Pembersihan Data Otomatis")

df = df_raw.copy()

# Menghapus kolom kosong seluruhnya
df.dropna(axis=1, how='all', inplace=True)

# Menghapus duplikasi
df.drop_duplicates(inplace=True)

# Membersihkan nama kolom (hapus spasi dan ubah ke huruf kecil)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Mengisi nilai kosong dengan median (untuk kolom numerik)
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

st.write("### Contoh Data Setelah Dibersihkan:")
st.dataframe(df.head())

# ============================================================
# 4. VISUALISASI SEDERHANA
# ============================================================
st.subheader("📊 Visualisasi Awal")

if "year" in df.columns and "access_clean_water" in df.columns:
    avg_access = df.groupby("year")["access_clean_water"].mean().reset_index()
    st.line_chart(avg_access, x="year", y="access_clean_water")
else:
    st.info("Kolom `year` atau `access_clean_water` tidak ditemukan di dataset.")

# ============================================================
# 5. LOAD MODEL & PREDIKSI
# ============================================================
st.subheader("🤖 Prediksi Akses Air Bersih")

model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error("Model belum ditemukan. Pastikan file `model.pkl` ada di folder proyek.")
    st.stop()

model = joblib.load(model_path)

# Input user
st.markdown("Masukkan data untuk memprediksi tingkat akses air bersih:")
col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("Tahun", min_value=2000, max_value=2030, value=2024)
with col2:
    population = st.number_input("Jumlah Penduduk (juta)", min_value=0.0, value=5.0)
with col3:
    piped_water = st.number_input("Persentase Akses Air Pipa (%)", min_value=0.0, max_value=100.0, value=50.0)

if st.button("Prediksi"):
    input_data = pd.DataFrame({
        "year": [year],
        "population": [population],
        "piped_water": [piped_water]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💧 Prediksi akses air bersih: **{prediction:.2f}%**")

# ============================================================
# 6. KESIMPULAN / PENJELASAN
# ============================================================
st.subheader("📘 Penjelasan")
st.markdown("""
Aplikasi ini membantu menganalisis **ketersediaan dan pemerataan akses air bersih** di Jawa Barat.  
Data mentah akan dibersihkan secara otomatis sebelum dianalisis, agar hasil lebih akurat.  
Model prediksi dapat digunakan untuk memperkirakan tren akses air bersih di masa depan berdasarkan data demografi dan infrastruktur.
""")
