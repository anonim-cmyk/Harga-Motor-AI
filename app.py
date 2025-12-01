import streamlit as st
import pandas as pd
from model_utils import load_model, score_with_risk

st.set_page_config(page_title="SPK Harga Motor", layout="wide")

st.title("🛵 SPK Prediksi Harga Motor Bekas + Analisis Risiko")

@st.cache_data
def load_cached_model():
    return load_model("model.pkl")

pipeline, metadata = load_cached_model()

numeric_cols = metadata["numeric_cols"]
categorical_cols = metadata["categorical_cols"]

with st.sidebar:
    st.header("📁 Input File")
    file = st.file_uploader("Upload CSV untuk prediksi batch", type=["csv"])
    run_batch = st.checkbox("Jalankan prediksi untuk seluruh CSV")

# ======================================
# MODE BATCH (CSV)
# ======================================
if file:
    df = pd.read_csv(file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # perbaikan kolom agar Arrow-compatible
    for c in df.columns:
        try:
            df[c] = df[c].astype(str).str.replace(",", "").str.replace(" ", "")
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except:
            pass

    if run_batch:
        st.success("Prediksi sedang diproses...")

        preds = pipeline.predict(df)
        df["predicted_price"] = preds

        st.subheader("Hasil Prediksi")
        st.dataframe(df)

        st.download_button(
            "Download CSV Hasil Prediksi",
            df.to_csv(index=False),
            file_name="predictions.csv"
        )

else:
    # ======================================
    # MODE MANUAL
    # ======================================
    st.subheader("🔧 Prediksi Manual")

    with st.form("manual_form"):
        inputs = {}

        for col in numeric_cols:
            inputs[col] = st.number_input(col, value=0.0)

        for col in categorical_cols:
            inputs[col] = st.text_input(col)

        claimed = st.number_input("Harga yang diklaim penjual (opsional)", value=0.0)
        submit = st.form_submit_button("Prediksi")

    if submit:
        data = pd.DataFrame([inputs])

        # convert numeric
        for c in numeric_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

        risk = score_with_risk(pipeline, metadata, data, claimed if claimed > 0 else None)

        st.metric("Prediksi Harga Motor", f"{risk['predicted_price']:,.0f} Rp")

        st.subheader("Analisis Risiko")
        st.json(risk)

        st.info("Rekomendasi:")
        if risk["risk_level"] == "High":
            st.write("⚠️ Perbedaan harga sangat besar. Risiko penipuan tinggi.")
        elif risk["risk_level"] == "Medium":
            st.write("⚠️ Harga agak berbeda. Perlu verifikasi lebih lanjut.")
        else:
            st.write("✅ Harga sesuai perkiraan model. Risiko rendah.")

st.caption("Model berlaku indikatif. Tetap lakukan pengecekan manual (dokumen/foto).")
