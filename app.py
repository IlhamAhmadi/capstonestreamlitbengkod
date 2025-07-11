import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")

st.title("Mari Kita Prediksi Tingkat Obesitasmu")
st.markdown("Nah kamu wajib ngisi data dibawah ini biar bisa tauu")

try:
    with open("model_bagus.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Gagal memuat model. Pastikan file model.pkl ada di folder yang sama.")
    st.stop()

# Label encoders
label_maps = {
    'Gender': ['Cewek', 'Cowok'],
    'family_history_with_overweight': ['Iya', 'Enggak'],
    'FAVC': ['Iya', 'Enggak'],
    'CAEC': ['Enggak', 'Kadang sih', 'Sering hehe', 'Selaluuuu'],
    'SMOKE': ['Anti rokok gess', 'Woiya jelass harus perasapan'],
    'SCC': ['Enggak', 'Iya'],
    'CALC': ['Enggak', 'Kadang sih', 'Sering hehe', 'Selaluuuu'],
    'MTRANS': ['Mobill', 'Sepeda', 'Motoran', 'Transportasi publik (BRT contohnya)', 'Anak sehat jalan kaki']
}

def encode_column(value, classes):
    le = LabelEncoder()
    le.fit(classes)
    return le.transform([value])[0]

# Input user
gender = st.selectbox('Jenis Kelamin', label_maps['Gender'])
age = st.slider('Usia', 10, 100, 25)
height = st.slider('Tinggi Badan (m)', 1.3, 2.2, 1.7)
weight = st.slider('Berat Badan (kg)', 30, 200, 70)
family_history = st.selectbox('Ada keluarga yg obesitas?', label_maps['family_history_with_overweight'])
favc = st.selectbox('Sering makan makanan tinggi kalori?', label_maps['FAVC'])
fcvc = st.slider('Sering makan sayur gak? (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah makan besar /hari berapa banyak? (1-4)', 1, 4, 3)
caec = st.selectbox('Ngemil?', label_maps['CAEC'])
smoke = st.selectbox('Merokok?', label_maps['SMOKE'])
scc = st.selectbox('Kontrol kalori gak?', label_maps['SCC'])
ch2o = st.slider('Konsumsi air harian berapa banyak? (1-3 liter)', 1, 3, 2)
faf = st.slider('Frekuensi aktivitas fisiknya seberapa sering? (0-3)', 0, 3, 1)
tue = st.slider('Waktu menatap layar /hari (0-2 jam)', 0, 2, 1)
calc = st.selectbox('Konsumsi alkohol?', label_maps['CALC'])
mtrans = st.selectbox('Transportasi utamanya apa?', label_maps['MTRANS'])

# Encode semua data
input_data = {
    'Gender': encode_column(gender, label_maps['Gender']),
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': encode_column(family_history, label_maps['family_history_with_overweight']),
    'FAVC': encode_column(favc, label_maps['FAVC']),
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': encode_column(caec, label_maps['CAEC']),
    'SMOKE': encode_column(smoke, label_maps['SMOKE']),
    'SCC': encode_column(scc, label_maps['SCC']),
    'CH2O': ch2o,
    'FAF': faf,
    'TUE': tue,
    'CALC': encode_column(calc, label_maps['CALC']),
    'MTRANS': encode_column(mtrans, label_maps['MTRANS'])
}

input_df = pd.DataFrame([input_data])

# Standarisasi kolom numerik
scaler = StandardScaler()
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])

if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediksi Tingkat Obesitas Anda: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
