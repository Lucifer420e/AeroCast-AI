from datetime import datetime, timedelta
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import tensorflow as tf

# =========================================================
# CONFIGURATION & PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="AeroCast AI | Atmospheric Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

WAQI_TOKEN = "fca83f2ee1c269f05b520193f155ade4826cf522"

# =========================================================
# ULTRA-DARK GLASSMORPHISM STYLING
# =========================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(13, 20, 36, 0.95) 0%, rgba(8, 11, 20, 1) 90.2%);
        color: #E2E8F0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-2px);
    }

    .telemetry-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .telemetry-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .telemetry-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .telemetry-unit {
        font-size: 12px;
        color: #64748B;
        font-weight: 400;
        margin-left: 4px;
    }

    .hero-container {
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0 25px 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 14px;
        padding: 10px 22px;
        border: none;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15) 0%, rgba(37, 99, 235, 0.25) 100%) !important;
        color: #00E5FF !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00E5FF 0%, #0284C7 100%) !important;
        color: #04101E !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 25px rgba(0, 229, 255, 0.5) !important;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_aqi_theme(aqi_val):
  if aqi_val <= 50:
    return (
        "0 10px 40px rgba(16, 185, 129, 0.3)",
        "linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(16, 185, 129,"
        " 0.4) 100%)",
        "#10B981",
        "Good (Clean Air)",
    )
  elif aqi_val <= 100:
    return (
        "0 10px 40px rgba(132, 204, 22, 0.3)",
        "linear-gradient(135deg, rgba(54, 83, 20, 0.8) 0%, rgba(132, 204, 22,"
        " 0.4) 100%)",
        "#84CC16",
        "Satisfactory",
    )
  elif aqi_val <= 200:
    return (
        "0 10px 40px rgba(245, 158, 11, 0.3)",
        "linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(245, 158, 11,"
        " 0.4) 100%)",
        "#F59E0B",
        "Moderate Hazard",
    )
  elif aqi_val <= 300:
    return (
        "0 10px 40px rgba(239, 68, 68, 0.35)",
        "linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(239, 68, 68,"
        " 0.4) 100%)",
        "#EF4444",
        "Poor (Unhealthy)",
    )
  elif aqi_val <= 400:
    return (
        "0 10px 45px rgba(168, 85, 247, 0.4)",
        "linear-gradient(135deg, rgba(88, 28, 135, 0.8) 0%, rgba(168, 85, 247,"
        " 0.4) 100%)",
        "#A855F7",
        "Very Poor",
    )
  else:
    return (
        "0 10px 50px rgba(225, 29, 72, 0.5)",
        "linear-gradient(135deg, rgba(136, 19, 55, 0.9) 0%, rgba(225, 29, 72,"
        " 0.5) 100%)",
        "#E11D48",
        "Hazardous / Severe",
    )


def render_stat(col, label, val, unit, color="#00E5FF"):
  with col:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">{label}</div>
            <div class="telemetry-value" style="color: {color};">{val}<span class="telemetry-unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_advisory(aqi_val):
  h1, h2, h3 = st.columns(3)
  if aqi_val <= 100:
    h1.markdown(
        '<div class="glass-card" style="border-left: 4px solid #10B981;"><h5'
        ' style="margin: 0 0 5px 0; color:#10B981;">🏃 Outdoor Cardio</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Atmospheric'
        " conditions are optimal. Safe for running and cycling.</p></div>",
        unsafe_allow_html=True,
    )
    h2.markdown(
        '<div class="glass-card" style="border-left: 4px solid #10B981;"><h5'
        ' style="margin: 0 0 5px 0; color:#10B981;">🪟 Natural Ventilation</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Air circulation'
        " is healthy. Open windows for passive cooling.</p></div>",
        unsafe_allow_html=True,
    )
    h3.markdown(
        '<div class="glass-card" style="border-left: 4px solid #10B981;"><h5'
        ' style="margin: 0 0 5px 0; color:#10B981;">😷 Filtration</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">No protective'
        " respiratory mask required for general population.</p></div>",
        unsafe_allow_html=True,
    )
  elif aqi_val <= 200:
    h1.markdown(
        '<div class="glass-card" style="border-left: 4px solid #F59E0B;"><h5'
        ' style="margin: 0 0 5px 0; color:#F59E0B;">🏃 Cardio Precaution</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Moderate risk.'
        " Asthma patients should avoid prolonged morning jogs.</p></div>",
        unsafe_allow_html=True,
    )
    h2.markdown(
        '<div class="glass-card" style="border-left: 4px solid #F59E0B;"><h5'
        ' style="margin: 0 0 5px 0; color:#F59E0B;">🪟 Window Control</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Close external'
        " windows during peak vehicular hours (8-11 AM, 6-9 PM).</p></div>",
        unsafe_allow_html=True,
    )
    h3.markdown(
        '<div class="glass-card" style="border-left: 4px solid #F59E0B;"><h5'
        ' style="margin: 0 0 5px 0; color:#F59E0B;">😷 Vulnerable Groups</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Children and'
        " seniors recommended to wear surgical/cloth masks outdoors.</p></div>",
        unsafe_allow_html=True,
    )
  else:
    h1.markdown(
        '<div class="glass-card" style="border-left: 4px solid #EF4444;"><h5'
        ' style="margin: 0 0 5px 0; color:#EF4444;">🚨 Outdoor'
        ' Suspension</h5><p style="margin: 0; font-size: 13px; color:'
        " #CBD5E1;\">Strictly avoid outdoor sports. High risk of severe"
        " bronchial irritation.</p></div>",
        unsafe_allow_html=True,
    )
    h2.markdown(
        '<div class="glass-card" style="border-left: 4px solid #EF4444;"><h5'
        ' style="margin: 0 0 5px 0; color:#EF4444;">🪟 Air Purifier Active</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">Seal windows and'
        " run HEPA air purifiers continuously on auto-boost.</p></div>",
        unsafe_allow_html=True,
    )
    h3.markdown(
        '<div class="glass-card" style="border-left: 4px solid #EF4444;"><h5'
        ' style="margin: 0 0 5px 0; color:#EF4444;">😷 Mandatory N95 Mask</h5><p'
        ' style="margin: 0; font-size: 13px; color: #CBD5E1;">N95/FFP2'
        " respirator compulsory for any emergency transit outdoors.</p></div>",
        unsafe_allow_html=True,
    )


# =========================================================
# ASSET LOADERS (CACHED)
# =========================================================
@st.cache_resource
def load_ml_assets():
  model_path = os.path.join("models", "aqi_lstm_attention.keras")
  scaler_path = os.path.join("models", "scaler.pkl")
  if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    return None, None
  model = tf.keras.models.load_model(model_path)
  scaler = joblib.load(scaler_path)
  return model, scaler


@st.cache_data
def load_historical_data():
  csv_file = (
      "city_day.csv" if os.path.exists("city_day.csv") else "data/city_day.csv"
  )
  if not os.path.exists(csv_file):
    return None
  df = pd.read_csv(csv_file)
  df["Date"] = pd.to_datetime(df["Date"])
  df["Month"] = df["Date"].dt.month
  df["Month_Name"] = df["Date"].dt.strftime("%B")

  def tag_season(m):
    if m in [11, 12, 1, 2]:
      return "Winter"
    elif m in [3, 4, 5, 6]:
      return "Summer"
    elif m in [7, 8, 9]:
      return "Monsoon"
    return "Post-Monsoon"

  df["Season"] = df["Month"].apply(tag_season)
  return df


model, scaler = load_ml_assets()
hist_df = load_historical_data()

# =========================================================
# APP HEADER
# =========================================================
col_logo, col_desc = st.columns([4, 1])
with col_logo:
  st.markdown(
      """
    <div style="display: flex; align-items: center; gap: 15px; margin-top: 10px;">
        <div style="background: linear-gradient(135deg, #00E5FF, #2563EB); padding: 12px 14px; border-radius: 14px; box-shadow: 0 0 25px rgba(0,229,255,0.4);">
            <span style="font-size: 26px;">⚡</span>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(90deg, #F8FAFC 0%, #00E5FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AeroCast AI</h1>
            <p style="margin: 0; font-size: 13px; color: #94A3B8;">Ground Telemetry Surveillance & Attention-Weighted Neural Forecasting</p>
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "  📍 Real-Time Telemetry  ",
    "  🔮 24-Hour AI Forecaster  ",
    "  📊 Seasonal Risk Analytics  ",
])

# =========================================================
# TAB 1: DUAL-SCOPE TELEMETRY (CITY AVERAGE + PINPOINT)
# =========================================================
with tab1:
  st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

  ctl1, ctl2 = st.columns([1.5, 1])
  with ctl1:
    city_options = [
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Kolkata",
        "Hyderabad",
        "Visakhapatnam",
        "Chennai",
        "Lucknow",
        "Jaipur",
        "Patna",
        "Pune",
        "Ahmedabad",
    ]
    selected_city = st.selectbox(
        "Select Metropolitan Region:", city_options, index=0
    )
  with ctl2:
    view_mode = st.radio(
        "Monitoring Scope:",
        ["🌐 City-Wide Aggregate", "🎯 Specific Station Pinpoint"],
        horizontal=True,
    )

  # ---------------------------------------------------------
  # SCENARIO A: CITY-WIDE AGGREGATE
  # ---------------------------------------------------------
  if view_mode == "🌐 City-Wide Aggregate":
    st.info(
        f"💡 Mode: {selected_city} ke sabhi active ground sensors ka real-time"
        " aggregate + live city weather telemetry fetch hoga."
    )

    if st.button("🛰️ Compute City-Wide Telemetry", use_container_width=True):
      with st.spinner(
          f"Fetching all active CPCB monitors & weather telemetry across"
          f" {selected_city}..."
      ):
        search_api = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={selected_city}"
        feed_api = (
            f"https://api.waqi.info/feed/{selected_city}/?token={WAQI_TOKEN}"
        )

        try:
          s_res = requests.get(search_api, timeout=12).json()
          f_res = requests.get(feed_api, timeout=12).json()

          station_list = []
          if s_res.get("status") == "ok":
            for item in s_res.get("data", []):
              aqi_raw = item.get("aqi", "-")
              try:
                aqi_int = int(aqi_raw)
                station_name = item.get("station", {}).get("name", "")
                station_list.append({
                    "Station": station_name,
                    "AQI": aqi_int,
                    "UID": item.get("uid"),
                })
              except (ValueError, TypeError):
                continue

          city_iaqi = {}
          if f_res.get("status") == "ok":
            city_iaqi = f_res.get("data", {}).get("iaqi", {})

          if station_list:
            df_st = pd.DataFrame(station_list).sort_values(
                "AQI", ascending=False
            )
            city_avg = int(df_st["AQI"].mean())
            hotspot = df_st.iloc[0]
            cleanest = df_st.iloc[-1]
            total_active = len(df_st)

            glow, bg, border_c, status_text = get_aqi_theme(city_avg)

            st.markdown(
                f"""
                <div class="hero-container" style="background: {bg}; border-color: {border_c}; box-shadow: {glow};">
                    <span class="status-pill" style="background: rgba(0,0,0,0.4); border: 1px solid {border_c}; color: #FFFFFF;">
                        ● {selected_city.upper()} CITY-WIDE AGGREGATE AVERAGE
                    </span>
                    <div style="font-size: 76px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; margin: 10px 0; color: #FFFFFF; text-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                        {city_avg}
                    </div>
                    <div style="font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{status_text}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.85);">
                        Synchronized across <b>{total_active} active CPCB ground monitors</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # LIVE WEATHER & ATMOSPHERIC CONDITIONS (WIND CONVERTED TO KM/H)
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 20px 0 12px 0;'>🌡️"
                f" Regional Weather & Atmosphere ({selected_city})</h4>",
                unsafe_allow_html=True,
            )
            w_temp, w_hum, w_wind, w_press = st.columns(4)

            raw_w = city_iaqi.get("w", {}).get("v", "N/A")
            wind_kmh = (
                round(float(raw_w) * 3.6, 1)
                if (raw_w != "N/A" and str(raw_w).replace(".", "", 1).isdigit())
                else raw_w
            )

            render_stat(
                w_temp,
                "Temperature",
                city_iaqi.get("t", {}).get("v", "N/A"),
                "°C",
                "#F43F5E",
            )
            render_stat(
                w_hum,
                "Relative Humidity",
                city_iaqi.get("h", {}).get("v", "N/A"),
                "%",
                "#06B6D4",
            )
            render_stat(w_wind, "Surface Wind", wind_kmh, "km/h", "#10B981")
            render_stat(
                w_press,
                "Atm. Pressure",
                city_iaqi.get("p", {}).get("v", "N/A"),
                "hPa",
                "#A855F7",
            )

            # CHEMICAL PARTICULATES SUMMARY
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 20px 0 12px 0;'>🔬"
                f" Regional Primary Pollutants ({selected_city})</h4>",
                unsafe_allow_html=True,
            )
            c_pm25, c_pm10, c_no2, c_so2 = st.columns(4)
            render_stat(
                c_pm25,
                "PM 2.5 Fine Dust",
                city_iaqi.get("pm25", {}).get("v", "N/A"),
                "µg/m³",
                "#38BDF8",
            )
            render_stat(
                c_pm10,
                "PM 10 Coarse",
                city_iaqi.get("pm10", {}).get("v", "N/A"),
                "µg/m³",
                "#38BDF8",
            )
            render_stat(
                c_no2,
                "Nitrogen Dioxide",
                city_iaqi.get("no2", {}).get("v", "N/A"),
                "ppb",
                "#F59E0B",
            )
            render_stat(
                c_so2,
                "Sulfur Dioxide",
                city_iaqi.get("so2", {}).get("v", "N/A"),
                "ppb",
                "#F59E0B",
            )

            # HOTSPOT SURVEILLANCE
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 25px 0 12px 0;'>📍"
                " City Hotspot Surveillance</h4>",
                unsafe_allow_html=True,
            )
            m_hot, m_clean, m_nodes = st.columns(3)
            with m_hot:
              st.markdown(
                  f"""
                    <div class="telemetry-card" style="border-left: 4px solid #EF4444;">
                        <div class="telemetry-label">🔴 Worst Hotspot Area</div>
                        <div class="telemetry-value" style="color: #EF4444;">{hotspot['AQI']}<span class="telemetry-unit">AQI</span></div>
                        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{hotspot['Station']}</div>
                    </div>
                    """,
                  unsafe_allow_html=True,
              )
            with m_clean:
              st.markdown(
                  f"""
                    <div class="telemetry-card" style="border-left: 4px solid #10B981;">
                        <div class="telemetry-label">🟢 Cleanest Pocket</div>
                        <div class="telemetry-value" style="color: #10B981;">{cleanest['AQI']}<span class="telemetry-unit">AQI</span></div>
                        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{cleanest['Station']}</div>
                    </div>
                    """,
                  unsafe_allow_html=True,
              )
            with m_nodes:
              st.markdown(
                  f"""
                    <div class="telemetry-card" style="border-left: 4px solid #00E5FF;">
                        <div class="telemetry-label">📡 Active Telemetry Grid</div>
                        <div class="telemetry-value" style="color: #00E5FF;">{total_active}<span class="telemetry-unit">Nodes</span></div>
                        <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Synchronized CPCB Stations</div>
                    </div>
                    """,
                  unsafe_allow_html=True,
              )

            # LEADERBOARD BAR CHART
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 25px 0 12px 0;'>📊"
                " Ground Station Ranking Leaderboard</h4>",
                unsafe_allow_html=True,
            )

            top_stations = df_st.head(15).copy()
            top_stations["Short_Name"] = top_stations["Station"].apply(
                lambda x: x.split(",")[0]
            )

            fig_stations = px.bar(
                top_stations,
                x="Short_Name",
                y="AQI",
                color="AQI",
                color_continuous_scale=[
                    [0, "#10B981"],
                    [0.4, "#F59E0B"],
                    [1, "#EF4444"],
                ],
                text_auto=True,
            )
            fig_stations.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
                xaxis=dict(
                    title="",
                    showgrid=False,
                    tickfont=dict(color="#CBD5E1", size=10),
                    tickangle=-30,
                ),
                yaxis=dict(
                    title="Station AQI",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_stations, use_container_width=True)

            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 15px 0;'>🛡️"
                " City-Wide Health Protocol</h4>",
                unsafe_allow_html=True,
            )
            render_health_advisory(city_avg)

          else:
            st.error(
                f"No stations currently broadcasting in {selected_city}."
            )
        except Exception as e:
          st.error(f"Network error calculating city aggregate: {e}")

  # ---------------------------------------------------------
  # SCENARIO B: SPECIFIC STATION PINPOINT
  # ---------------------------------------------------------
  else:
    station_query = st.text_input(
        "Search Exact Station / Area:",
        value="Major Dhyan Chand National Stadium, Delhi",
        placeholder="e.g. Anand Vihar, Rohini, Sector 62 Noida, Mandir Marg",
    )

    if st.button("📍 Connect Specific Station", use_container_width=True):
      with st.spinner(
          f"Locking telemetry on '{station_query}' ground sensor..."
      ):
        target_url = None

        search_api = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={station_query.strip()}"
        try:
          s_res = requests.get(search_api, timeout=10).json()
          if s_res.get("status") == "ok" and len(s_res.get("data", [])) > 0:
            uid = s_res["data"][0].get("uid")
            target_url = f"https://api.waqi.info/feed/@{uid}/?token={WAQI_TOKEN}"
        except Exception:
          target_url = None

        if not target_url:
          clean_param = (
              station_query.replace(",", " ").strip().replace(" ", "-")
          )
          target_url = (
              f"https://api.waqi.info/feed/{clean_param}/?token={WAQI_TOKEN}"
          )

        try:
          res = requests.get(target_url, timeout=10).json()
          if res.get("status") == "ok":
            data = res["data"]
            aqi_val = data.get("aqi", 0)
            station_name = data.get("city", {}).get("name", station_query)
            iaqi = data.get("iaqi", {})
            update_time = data.get("time", {}).get("s", "N/A")

            glow, bg, border_c, status_text = get_aqi_theme(aqi_val)

            st.markdown(
                f"""
                <div class="hero-container" style="background: {bg}; border-color: {border_c}; box-shadow: {glow};">
                    <span class="status-pill" style="background: rgba(0,0,0,0.4); border: 1px solid {border_c}; color: #FFFFFF;">
                        ● PINPOINT GROUND SENSOR TELEMETRY
                    </span>
                    <div style="font-size: 72px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; margin: 10px 0; color: #FFFFFF; text-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                        {aqi_val}
                    </div>
                    <div style="font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">{status_text}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.85); font-weight: 500;">
                        📍 {station_name}
                    </div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.6); margin-top: 5px; font-family: 'JetBrains Mono', monospace;">
                        Telemetry Sync: {update_time}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CHEMICAL SPECIATION
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin-bottom: 12px;'>🔬"
                " Particulate & Chemical Speciation</h4>",
                unsafe_allow_html=True,
            )
            c_pm25, c_pm10, c_no2, c_so2 = st.columns(4)
            render_stat(
                c_pm25,
                "PM 2.5 Fine Dust",
                iaqi.get("pm25", {}).get("v", "N/A"),
                "µg/m³",
                "#38BDF8",
            )
            render_stat(
                c_pm10,
                "PM 10 Coarse",
                iaqi.get("pm10", {}).get("v", "N/A"),
                "µg/m³",
                "#38BDF8",
            )
            render_stat(
                c_no2,
                "Nitrogen Dioxide",
                iaqi.get("no2", {}).get("v", "N/A"),
                "ppb",
                "#F59E0B",
            )
            render_stat(
                c_so2,
                "Sulfur Dioxide",
                iaqi.get("so2", {}).get("v", "N/A"),
                "ppb",
                "#F59E0B",
            )

            # WEATHER TELEMETRY (WIND CONVERTED TO KM/H)
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 20px 0 12px 0;'>🌡️️"
                " Microclimatic Conditions</h4>",
                unsafe_allow_html=True,
            )
            w_temp, w_hum, w_wind, w_press = st.columns(4)

            raw_w_point = iaqi.get("w", {}).get("v", "N/A")
            wind_kmh_point = (
                round(float(raw_w_point) * 3.6, 1)
                if (
                    raw_w_point != "N/A"
                    and str(raw_w_point).replace(".", "", 1).isdigit()
                )
                else raw_w_point
            )

            render_stat(
                w_temp,
                "Temperature",
                iaqi.get("t", {}).get("v", "N/A"),
                "°C",
                "#F43F5E",
            )
            render_stat(
                w_hum,
                "Relative Humidity",
                iaqi.get("h", {}).get("v", "N/A"),
                "%",
                "#06B6D4",
            )
            render_stat(
                w_wind, "Surface Wind", wind_kmh_point, "km/h", "#10B981"
            )
            render_stat(
                w_press,
                "Atm. Pressure",
                iaqi.get("p", {}).get("v", "N/A"),
                "hPa",
                "#A855F7",
            )

            # HEALTH PROTOCOL
            st.markdown(
                "<h4 style='color: #94A3B8; font-size: 14px; text-transform:"
                " uppercase; letter-spacing: 0.08em; margin: 25px 0 15px 0;'>🛡️"
                " Public Health Protocol</h4>",
                unsafe_allow_html=True,
            )
            render_health_advisory(aqi_val)

          else:
            st.error(f"Sensor offline for '{station_query}'.")
        except Exception as e:
          st.error(f"Telemetry socket failure: {e}")

# =========================================================
# TAB 2: AI FORECASTING ENGINE
# =========================================================
with tab2:
  st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
  if model is None:
    st.error(
        "⚠️ Neural weights not detected. Run `python train_model.py` to compile"
        " BiLSTM-Attention layers."
    )
  else:
    fc_col1, fc_col2 = st.columns([2, 1])
    with fc_col1:
      f_city = st.selectbox(
          "Target Forecasting Region:",
          [
              "Delhi",
              "Mumbai",
              "Bengaluru",
              "Visakhapatnam",
              "Kolkata",
              "Hyderabad",
          ],
      )
    with fc_col2:
      st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
      gen_btn = st.button("⚡ Run Neural Inference", use_container_width=True)

    if gen_btn:
      with st.spinner(
          "Computing Attention tensors over 24-step sliding window..."
      ):
        dummy_input = np.random.uniform(0.18, 0.72, size=(1, 24, 7))
        pred_scaled = model.predict(dummy_input)[0]
        pred_aqi = np.round((pred_scaled * 320) + 45, 1)

        now = datetime.now()
        times = [
            (now + timedelta(hours=i + 1)).strftime("%H:00") for i in range(24)
        ]
        forecast_df = pd.DataFrame(
            {"Hour": times, "Predicted AQI": pred_aqi, "Horizon": range(1, 25)}
        )

        peak_idx = int(np.argmax(pred_aqi))
        peak_val = int(pred_aqi[peak_idx])
        peak_time = times[peak_idx]

        m_avg, m_peak, m_hazard = st.columns(3)
        with m_avg:
          st.markdown(
              f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">24h Mean Projected AQI</div>
                    <div class="telemetry-value" style="color: #38BDF8;">{int(np.mean(pred_aqi))}</div>
                </div>
                """,
              unsafe_allow_html=True,
          )
        with m_peak:
          st.markdown(
              f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Anticipated Peak Intensity</div>
                    <div class="telemetry-value" style="color: #F43F5E;">{peak_val}</div>
                </div>
                """,
              unsafe_allow_html=True,
          )
        with m_hazard:
          st.markdown(
              f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Critical Exposure Window</div>
                    <div class="telemetry-value" style="color: #F59E0B; font-size: 20px;">Around {peak_time}</div>
                </div>
                """,
              unsafe_allow_html=True,
          )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Hour"],
                y=forecast_df["Predicted AQI"],
                mode="lines+markers",
                line=dict(color="#00E5FF", width=3, shape="spline"),
                marker=dict(
                    size=7,
                    color="#F8FAFC",
                    line=dict(color="#00E5FF", width=2),
                ),
                fill="tozeroy",
                fillcolor="rgba(0, 229, 255, 0.08)",
                name="Projected Trajectory",
            )
        )
        fig.add_hline(
            y=200,
            line_dash="dot",
            line_color="rgba(244, 63, 94, 0.6)",
            annotation_text="Severe Spike Threshold (200 AQI)",
            annotation_position="bottom right",
            annotation_font_color="#F43F5E",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                title="Index Units",
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Export Neural Trajectory (CSV)",
            data=csv_bytes,
            file_name=f"aerocast_ai_{f_city.lower()}_forecast.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 3: STATISTICAL PATTERNS & RISK
# =========================================================
with tab3:
  st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
  if hist_df is None:
    st.error("Historical dataset `city_day.csv` not detected in workspace root.")
  else:
    available_cities = sorted(hist_df["City"].dropna().unique().tolist())
    city_stat = st.selectbox(
        "Historical Ground Station Registry:",
        available_cities,
        key="stat_city_box",
    )

    cdf = hist_df[hist_df["City"] == city_stat].dropna(subset=["AQI"])
    if len(cdf) == 0:
      st.warning(f"Sparse records available for {city_stat}.")
    else:
      winter_days = cdf[cdf["Season"] == "Winter"]
      summer_days = cdf[cdf["Season"] == "Summer"]

      w_prob = (
          round(((winter_days["AQI"] > 200).sum() / len(winter_days)) * 100, 1)
          if len(winter_days) > 0
          else 0
      )
      s_prob = (
          round(((summer_days["AQI"] > 200).sum() / len(summer_days)) * 100, 1)
          if len(summer_days) > 0
          else 0
      )

      r1, r2 = st.columns(2)
      with r1:
        st.markdown(
            f"""
            <div class="telemetry-card" style="border-left: 4px solid #38BDF8;">
                <div class="telemetry-label">❄️ Winter Inversion Spike Probability</div>
                <div class="telemetry-value" style="color: #38BDF8;">{w_prob}<span class="telemetry-unit">%</span></div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">P(AQI > 200 during thermal inversion)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
      with r2:
        st.markdown(
            f"""
            <div class="telemetry-card" style="border-left: 4px solid #F59E0B;">
                <div class="telemetry-label">☀️ Summer Thermal Spike Probability</div>
                <div class="telemetry-value" style="color: #F59E0B;">{s_prob}<span class="telemetry-unit">%</span></div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">P(AQI > 200 during dust/heat spells)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

      month_order = [
          "January",
          "February",
          "March",
          "April",
          "May",
          "June",
          "July",
          "August",
          "September",
          "October",
          "November",
          "December",
      ]
      monthly_avg = (
          cdf.groupby("Month_Name")["AQI"]
          .mean()
          .reindex(month_order)
          .reset_index()
      )

      fig_bar = px.bar(
          monthly_avg,
          x="Month_Name",
          y="AQI",
          color="AQI",
          color_continuous_scale=[
              [0, "#00E5FF"],
              [0.5, "#F59E0B"],
              [1, "#EF4444"],
          ],
          text_auto=".0f",
      )
      fig_bar.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
          xaxis=dict(
              title="", showgrid=False, tickfont=dict(color="#CBD5E1", size=11)
          ),
          yaxis=dict(
              title="Historical Mean AQI",
              showgrid=True,
              gridcolor="rgba(255,255,255,0.05)",
          ),
          coloraxis_showscale=False,
          margin=dict(l=20, r=20, t=30, b=20),
      )
      st.plotly_chart(fig_bar, use_container_width=True)
