import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from streamlit_option_menu import option_menu
import geopandas as gpd
import matplotlib as mpl
import io
import base64
import joblib
import gdown
import os

# -----------------------------
st.set_page_config("Chicago Crime Dashboard", "👮", layout="wide")

# -----------------------------
# Define GDrive download helper
@st.cache_resource
def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# File IDs from Google Drive
GDRIVE_FILES = {
    "model_data_ready.csv": "1Xpc0nxQp0BMLfN4xvG4CmGwm7O1MTMu4",
    "filtered_data.csv": "1sBj3AvpPibqWKKHq-r0HOmfmrtTuKyZH",
    "model_rf.pkl": "1IQUSMDlP5M6pltUu1Nkec-92583FIvNy"
}

# Download files only if not exist
for filename, file_id in GDRIVE_FILES.items():
    download_from_gdrive(file_id, filename)

# -----------------------------
# USE LOCAL & DOWNLOADED FILES
DATA_PATH = "data/"

@st.cache_data
def load_eda_data():
    return {
        "monthly": pd.read_csv(DATA_PATH + "eda_summary.csv"),
        "top10": pd.read_csv(DATA_PATH + "top_10_crimes.csv"),
        "hourly": pd.read_csv(DATA_PATH + "hourly_crime.csv"),
        "weekday": pd.read_csv(DATA_PATH + "weekday_crime.csv"),
        "heatmap": pd.read_csv(DATA_PATH + "heatmap_data.csv"),
        "geo": pd.read_csv(DATA_PATH + "geo_summary.csv")
    }

@st.cache_data
def load_reference():
    ca = pd.read_csv(DATA_PATH + "community_area_ref.csv")
    return ca.rename(columns={"AREA_NUMBE": "community_area", "COMMUNITY": "area_name"})

@st.cache_data
def load_geojson():
    with open(DATA_PATH + "chicago_community_area.geojson", "r") as f:
        return json.load(f)

@st.cache_data
def load_model_data():
    return pd.read_csv("model_data_ready.csv")  # from GDrive root

@st.cache_resource
def load_model(model_name):
    return joblib.load(model_name)  # from GDrive root

@st.cache_resource
def load_encoder():
    return joblib.load(DATA_PATH + "encoders.pkl")

@st.cache_data
def load_historical_lookup():
    return pd.read_csv("data/historical_lookup.csv")

loading_placeholder = st.empty()

with loading_placeholder.container():
    st.info("🔄 Loading dashboard data... please wait...")

    eda_data = load_eda_data()
    ca_df = load_reference()
    geojson_data = load_geojson()
    df_model = load_model_data()
    df_raw = load_raw_data()
    historical_lookup = load_historical_lookup()

loading_placeholder.empty()  # Hapus pesan loading setelah semua selesai

# -----------------------------
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar .nav-link {
                font-size: 14px !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Remove top icon/label by setting title to empty string
    menu = option_menu(
        menu_title="",  # This removes the 'Main Menu' label
        options=["Home", "Model Prediction"],
        default_index=0
    )   

    st.markdown("""
    <hr style='margin-top:30px;margin-bottom:10px;'>
    <p style='font-size:13px; color: gray;'>
    <strong>Created by:</strong><br>
    Natali Valentina Sutanto<br>
    Data Science, Binus University<br>
    Skripsi 2024/2025
    </p>
    """, unsafe_allow_html=True)

# -----------------------------
# HOME: VISUALISASI EDA
# -----------------------------
if menu == "Home":
    st.markdown("""
    <h3 style='margin-top: 0; margin-bottom: 0;'>Chicago Crime - Dashboard EDA</h3>
    <p style='font-size: 19px; margin-top: 0;'>Analisis umum terhadap tren kejahatan berdasarkan waktu, lokasi, dan jenis kasus</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='font-size:18px; font-weight:600; margin:5px 0;'>1. Rata-rata Kejahatan Bulanan per Tahun</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        sns.barplot(data=eda_data["monthly"], x="year", y="crimes_per_month", ax=ax)
        ax.tick_params(axis='x', labelrotation=40, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        st.pyplot(fig)

    with col2:
        st.markdown("<p style='font-size:18px; font-weight:600; margin:5px 0;'>2. 10 Jenis Kejahatan Terbanyak</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.tick_params(axis='y', labelsize=9)
        sns.barplot(data=eda_data["top10"], y="primary_type", x="count", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<p style='font-size:18px; font-weight:600; margin:5px 0;'>3. Distribusi Jam Kejadian</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=eda_data["hourly"], x="hour", y="count", ax=ax)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        st.pyplot(fig)

    with col4:
        st.markdown("<p style='font-size:18px; font-weight:600; margin:5px 0;'>4. Distribusi Hari Kejadian</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=eda_data["weekday"], x="weekday", y="count", ax=ax)
        ax.tick_params(axis='x', labelrotation=35, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        st.pyplot(fig)

    # Plot 5 – Heatmap per Jenis Kejahatan
    st.markdown("""
    <p style='font-size:18px; font-weight:600; margin-bottom:8px;'>5. Heatmap per Jenis Kejahatan</p>
    <p style='font-size:16px; margin-top:-4px; margin-bottom:0px;'>Pilih jenis kejahatan:</p>
    
    <style>
    /* Hilangkan margin atas dari selectbox */
    section[data-testid="stSelectbox"] > div {
        margin-top: -50px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    crimes = eda_data["heatmap"]['primary_type'].unique()
    selected = st.selectbox("", crimes)

    st.markdown("""
        <style>
        .tight-label {
            font-size: 18px;
            font-weight: normal;
            margin-top:0px;
            margin-bottom: 0px;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)

    pivot = eda_data["heatmap"]
    subset = pivot[pivot['primary_type'] == selected].pivot(index="hour", columns="weekday", values="count").fillna(0)

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    subset = subset[weekday_order]

    mask = np.full(subset.shape, False)
    indices = np.random.choice(subset.size, size=int(subset.size * 0.3), replace=False)
    mask.flat[indices] = True
    annot_matrix = subset.where(mask).round(0).astype("Int64").astype(str)
    annot_matrix = annot_matrix.where(mask, '')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        subset,
        cmap="YlGnBu",
        ax=ax,
        annot=annot_matrix,
        fmt='',
        annot_kws={"size": 8},
        linewidths=0.2,
        linecolor='gray'
    )

    ax.set_xlabel("Hari", fontsize=8)
    ax.set_ylabel("Jam", fontsize=8)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(top=0.88, bottom=0.20)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(buf)

    # Peta - Visualisasi 6
    st.markdown("<p style='font-size:18px; font-weight:600; margin:5px 0;'>6. Peta Jumlah Kasus per Community Area</p>", unsafe_allow_html=True)

    try:
        # Load GeoJSON
        gdf_ca = gpd.read_file("data/chicago_community_area.geojson")
        gdf_ca = gdf_ca.rename(columns={"area_numbe": "community_area"})
        gdf_ca["community_area"] = gdf_ca["community_area"].astype(int)

        # Load crime data (from EDA preprocessed data)
        geo_df = eda_data["geo"][["community_area", "total_cases"]]

        # Merge
        merged = gdf_ca.merge(geo_df, on="community_area", how="left").fillna(0)
        merged = merged.set_geometry("geometry")

        # Setup colorbar & plot
        vmin = merged['total_cases'].min()
        vmax = merged['total_cases'].max()
        cmap = mpl.cm.Reds
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        fig_map, ax_map = plt.subplots(figsize=(9, 5))
        merged.plot(column='total_cases',
                    cmap=cmap,
                    linewidth=0.5,
                    edgecolor='black',
                    ax=ax_map)

        ax_map.tick_params(axis='both', labelsize=9)

        # Colorbar manual
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig_map.colorbar(sm, ax=ax_map, orientation="vertical", shrink=0.6)
        cbar.ax.tick_params(labelsize=9)

        # Export to buffer and display centered
        buf = io.BytesIO()
        fig_map.savefig(buf, format="png", bbox_inches='tight', dpi=120)
        buf.seek(0)

        img_bytes = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f"<div style='text-align: center;'><img src='data:image/png;base64,{img_bytes}' width='450'/></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Gagal menampilkan peta. Pastikan file 'chicago_community_area.geojson' valid dan sesuai.")
        st.text(str(e))
        
month_dict = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# =========================
# MODEL PREDICTION
# =========================
if menu == "Model Prediction":
    with st.spinner("🔍 Loading prediction page... please wait."):
        st.markdown("""
        <h3 style='margin-top: 0;'>Prediksi Jumlah Kasus Kriminal dengan Machine Learning </h3>
        <p style='font-size: 20px; margin-top: 0;'><strong>Perbandingan implementasi algoritma: Random Forest vs XGBoost vs LightGBM</strong></p>
        <p style='font-size: 19px; margin-top: 0;'>Silakan pilih filter input di sebelah kiri untuk melihat hasil prediksi berdasarkan algoritma</p>
        """, unsafe_allow_html=True)
    
        encoder = load_encoder()
    
        # Mapping bulan angka ke nama
        month_dict = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
    
        # --- Sidebar Filters ---
        with st.sidebar:
            st.markdown("### 🎯 Filter Input")
            available_years = list(range(2020, 2031))
            available_months = list(range(1, 13))
    
            selected_year = st.selectbox('Select a Year', sorted(available_years, reverse=True))
            month_display = [month_dict[m] for m in available_months]
            selected_month_display = st.selectbox('Select a Month', month_display)
            selected_month = [k for k, v in month_dict.items() if v == selected_month_display][0]
    
            selected_crime = st.selectbox('Select Crime Type', sorted(df_model['primary_type'].dropna().unique()))
            area_df = df_model[['community_area', 'area_name']].drop_duplicates().sort_values('area_name')
            selected_area_name = st.selectbox("Select Community Area", area_df['area_name'])
            selected_area = area_df[area_df['area_name'] == selected_area_name]['community_area'].values[0]
            selected_algo = st.selectbox('Select Algorithm', ['Random Forest', 'XGBoost', 'LightGBM'])
    
        # --- Load encoder & model ---
        encoders = joblib.load("data/encoders.pkl")
    
        df_filtered = df_raw[
            (df_raw['year'] == selected_year) &
            (df_raw['month'] == selected_month) &
            (df_raw['community_area'] == selected_area) &
            (df_raw['primary_type'] == selected_crime)
        ]
    
        # Contextual Feature: Historical or synthetic if future
        future_year = selected_year > df_model['year'].max()
        if not df_filtered.empty and not future_year:
            arrest_rate = round(df_filtered['arrest'].mean() * 100, 1)
            domestic_rate = round(df_filtered['domestic'].mean() * 100, 1)
            peak_hour = df_filtered['hour'].mode()[0]
            hour_bins = pd.cut(df_filtered['hour'], bins=[0, 6, 12, 18, 24],
                               labels=["12AM–6AM", "6AM–12PM", "12PM–6PM", "6PM–12AM"])
            peak_time = hour_bins.value_counts().idxmax()
            top_locations = df_filtered['location_description'].value_counts().head(3).index.tolist()
        else:
            lookup_row = historical_lookup[
                (historical_lookup['community_area'] == selected_area) &
                (historical_lookup['primary_type'] == selected_crime)
            ]
            
            if not lookup_row.empty:
                arrest_rate = round(lookup_row['arrest'].values[0] * 100, 1)
                domestic_rate = round(lookup_row['domestic'].values[0] * 100, 1)
                peak_hour = int(lookup_row['hour'].values[0])
                peak_time = pd.cut([peak_hour], bins=[0, 6, 12, 18, 24],
                                   labels=["12AM–6AM", "6AM–12PM", "12PM–6PM", "6PM–12AM"])[0]
                top_locations = ["N/A"]
            else:
                arrest_rate = 15.0
                domestic_rate = 8.0
                peak_hour = 12
                peak_time = "12PM–6PM"
                top_locations = ["N/A"]
            
                # Prepare input for prediction
                input_df = pd.DataFrame([{
                    'year': selected_year,
                    'month': selected_month,
                    'community_area': selected_area,
                    'primary_type_enc': encoders['primary_type'].transform([selected_crime])[0],
                    'arrest': arrest_rate / 100,
                    'domestic': domestic_rate / 100,
                    'hour': peak_hour
                }])
        
        # Predict
        if selected_algo == 'Random Forest':
            try:
                model = load_model("model_rf.pkl")
                log_pred = model.predict(input_df)[0]
                pred_cases = int(np.round(np.expm1(log_pred)))
    
            except Exception as e:
                st.error("❌ Error saat memprediksi")
                st.text(str(e))
        elif selected_algo == 'XGBoost':
            model = joblib.load("data/model_xgb.pkl")
        elif selected_algo == 'LightGBM':
            model = joblib.load("data/model_lgbm.pkl")

        prediction_log = model.predict(input_df)[0]
        predicted_cases = int(round(np.expm1(prediction_log)))
    
        # 5-Year Historical Comparison
        past_5yr = df_model[
            (df_model['year'] >= selected_year - 5) & (df_model['year'] < selected_year) &
            (df_model['month'] == selected_month) &
            (df_model['community_area'] == selected_area) &
            (df_model['primary_type'] == selected_crime)
        ]
        avg_5yr = int(past_5yr['case_count'].mean()) if not past_5yr.empty else 0
        pct_change = round(((predicted_cases - avg_5yr) / avg_5yr) * 100, 1) if avg_5yr > 0 else 0
        change_icon = "🔺" if pct_change > 0 else "🔻" if pct_change < 0 else "➖"
    
        # Model Metrics (manual)
        model_metrics = {
            "Random Forest": {"mae": 3.62, "smape": 27.75, "r2": 0.93},
            "XGBoost": {"mae": 4.75, "smape": 33.25, "r2": 0.87},
            "LightGBM": {"mae": 6.07, "smape": 38.13, "r2": 0.77}
        }
        metrics = model_metrics[selected_algo]
        mae, smape, r2 = metrics['mae'], metrics['smape'], metrics['r2']
        fit_text = "Strong fit" if r2 >= 0.85 else "Moderate fit" if r2 >= 0.7 else "Weak fit"
    
        # Display Output
        st.markdown("<h4 style='margin-top: 20px;'>🔎 Ringkasan Prediksi</h4>", unsafe_allow_html=True)
    
        # Location and Algo Info
        st.markdown(f"""
        <div style='font-size:19px; line-height:1.6;'>
        Prediksi untuk <b>{selected_area_name}</b>, bulan <b>{selected_month_display}</b> tahun <b>{selected_year}</b><br>
        <b>Jenis Kejahatan:</b> {selected_crime.upper()}<br>
        <b>Algoritma:</b> {selected_algo}
        </div>
        """, unsafe_allow_html=True)
    
        # Total Predicted Cases
        st.markdown(f"""
        <div style='
            background-color:#ffe3e3;
            padding:15px;
            border-radius:10px;
            font-size:18px;
            color:#c92a2a;
            font-weight:500;
            margin-top:20px;
            margin-bottom:20px;
        '>
            Prediksi: <strong>{predicted_cases} kasus</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Others Summary
        st.markdown(f"""
        <div style='font-size:19px; line-height:1.8;'>
        <b>● Rata-rata 5 Tahun Terakhir ({selected_month_display}):</b> {avg_5yr} → {change_icon} <b>{abs(pct_change)}%</b><br>
        <b>● Lokasi yang Paling Banyak Terjadi Kasus:</b> {', '.join(top_locations) if top_locations else 'N/A'}<br>
        <b>● Waktu Rawan:</b> Pukul {peak_time}<br>
        <b>● Persentase Tingkat Penangkapan:</b> {arrest_rate}%<br>
        <b>● Persentase Kasus Domestik:</b> {domestic_rate}%
        </div>
        """, unsafe_allow_html=True)

    # Model Evaluation
    st.markdown(f"<h4 style='margin-top: 25px; font-weight:normal;'>Hasil Evaluasi Model: <span style='font-weight:bold'>{selected_algo}</span></h4>", unsafe_allow_html=True)

    # Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background-color:#f9f9fa; padding:18px; border-radius:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <p style='margin:0; font-size:18px; color:#6c757d;'>MAE</p>
            <p style='margin:0; font-size:24px; font-weight:bold; color:#ff6b6b;'>{mae:.2f} cases</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background-color:#f9f9fa; padding:18px; border-radius:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <p style='margin:0; font-size:18px; color:#6c757d;'>RMSE</p>
            <p style='margin:0; font-size:24px; font-weight:bold; color:#ffa94d;'>{rmse:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background-color:#f9f9fa; padding:18px; border-radius:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <p style='margin:0; font-size:18px; color:#6c757d;'>R² Score</p>
            <p style='margin:0; font-size:24px; font-weight:bold; color:#51cf66;'>{r2:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Bar Chart
    st.markdown(f"<h4 style='margin-top: 24px;'>Prediksi vs Rata-rata 5 Tahun</span></h4>", unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Kategori": ["Prediksi", "Rata-rata 5 Tahun"],
        "Jumlah Kasus": [predicted_cases, avg_5yr]
    })
    fig_bar = px.bar(comparison_df, x="Kategori", y="Jumlah Kasus", color="Kategori", color_discrete_sequence=["#1f77b4", "#ff7f0e"])
    st.plotly_chart(fig_bar, use_container_width=True)
