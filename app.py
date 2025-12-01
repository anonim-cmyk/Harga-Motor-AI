# app.py
import streamlit as st
import pandas as pd
from model_utils import load_model, predict_single, score_with_risk
import joblib

st.set_page_config(page_title="SPK Harga Motor Bekas", layout="wide")

st.title("SPK Prediksi Harga Motor Bekas + Analisis Risiko")

@st.cache_data(ttl=300)
def load_model_cached(path="model.pkl"):
    return load_model(path)

with st.sidebar:
    st.header("Kontrol")
    model_path = st.text_input("Path model .pkl", value="model.pkl")
    uploaded = st.file_uploader("Upload CSV data (opsional)", type=["csv"])
    st.markdown("---")
    run_train = st.checkbox("Saya ingin menjalankan prediksi pada seluruh CSV yang diupload", value=False)
    st.markdown("File training harus dijalankan terlebih dahulu (lihat README).")

pipeline = None
metadata = None
try:
    pipeline, metadata = load_model_cached(model_path)
except Exception as e:
    st.sidebar.error(f"Gagal memuat model dari {model_path}: {e}")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview data yang diupload")
    st.dataframe(df.head())

    if run_train:
        st.info("Menjalankan prediksi untuk semua baris pada file...")
        preds = pipeline.predict(df)
        df["predicted_price"] = preds
        st.subheader("Hasil Prediksi")
        st.dataframe(df.head(100))
        csv = df.to_csv(index=False)
        st.download_button("Download hasil prediksi CSV", csv, file_name="predictions.csv")
else:
    st.subheader("Prediksi manual (isi form)")
    # Build form based on metadata columns if available
    if metadata:
        numeric_cols = metadata.get("numeric_cols", [])
        categorical_cols = metadata.get("categorical_cols", [])
    else:
        numeric_cols = ["year", "km", "engine_cc"]
        categorical_cols = ["brand", "model", "transmission", "fuel"]

    st.write("Isi fitur berikut (form adapts to kolom yang tersedia pada model).")
    form = st.form("pred_form")
    inputs = {}
    for col in numeric_cols:
        inputs[col] = form.number_input(col, value=0)
    for col in categorical_cols:
        inputs[col] = form.text_input(col, value="")
    claimed_price = form.number_input("Harga yang diklaim penjual (opsional)", value=0.0)
    submitted = form.form_submit_button("Prediksi & Analisis Risiko")
    if submitted:
        input_df = pd.DataFrame([inputs])
        # cast numeric columns to numeric
        for c in numeric_cols:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0)
        st.write("Input:", input_df.T)
        pred = float(pipeline.predict(input_df)[0])
        st.metric("Prediksi Harga (Rp)", f"{pred:,.0f}")
        risk = score_with_risk(pipeline, metadata, input_df, claimed_price if claimed_price>0 else None)
        st.write("Hasil Analisis Risiko:")
        st.json(risk)

        st.info("Rekomendasi singkat:")
        if risk["risk_level"] == "High":
            st.write("- Harga klaim jauh dari prediksi. Hati-hati: kemungkinan penipuan atau data keliru.")
        elif risk["risk_level"] == "Medium":
            st.write("- Harga sedikit berbeda. Perlu verifikasi lebih lanjut (foto, dokumen).")
        else:
            st.write("- Risiko rendah. Harga sesuai ekspektasi model.")

st.markdown("---")
st.caption("Catatan: model dan risk scoring bersifat indikatif. Gunakan verifikasi manual (dokumen/foto/test drive).")
