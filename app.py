import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import time
from datetime import datetime
from pathlib import Path
from src.features import build_features
from src.scoring import apply_policy_A, apply_policy_B

# Page Configuration
st.set_page_config(page_title="Ali Quant | BIST100", layout="wide", initial_sidebar_state="collapsed")

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=900)
def get_market_data():
    tickers = [
        "AEFES.IS", "AGHOL.IS", "AGROT.IS", "AHGAZ.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", 
        "ALFAS.IS", "ALTNY.IS", "ANHYT.IS", "ANSGR.IS", "ARCLK.IS", "ARDYZ.IS", "ASELS.IS", "ASTOR.IS", 
        "AVPGY.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "CANTE.IS", 
        "CCOLA.IS", "CIMSA.IS", "CLEBI.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "EFOR.IS", 
        "EGEEN.IS", "EKGYO.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FROTO.IS", 
        "GARAN.IS", "GESAN.IS", "GOLTS.IS", "GRTHO.IS", "GSRAY.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", 
        "IEYHO.IS", "ISCTR.IS", "ISMEN.IS", "KARSN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KONYA.IS", 
        "KRDMD.IS", "KTLEV.IS", "LMKDC.IS", "MAGEN.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS", "MPARK.IS", 
        "OBAMS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PASEU.IS", "PETKM.IS", "PGSUS.IS", "RALYH.IS", 
        "REEDR.IS", "RYGYO.IS", "SAHOL.IS", "SASA.IS", "SELEC.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", 
        "SOKM.IS", "TABGD.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TRALT.IS", 
        "TRENJ.IS", "TRMET.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS", 
        "VAKBN.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS"
    ]
    data = yf.download(tickers, period="6mo", group_by="ticker", progress=False)
    rows = []
    for t in tickers:
        try:
            temp = data[t].reset_index()
            if temp.empty: continue
            temp["ticker"] = t.replace(".IS", "")
            rows.append(temp)
        except: continue
    
    df = pd.concat(rows, ignore_index=True)
    df.rename(columns={"Date": "TRADE DATE", "Close": "CLOSING PRICE"}, inplace=True)
    for c in ["Open", "High", "Low"]:
        df[f"{c.upper()} PRICE"] = df[c]
    df["TRADE DATE"] = pd.to_datetime(df["TRADE DATE"]).dt.tz_localize(None)
    return df

def save_snapshot(df):
    """Saves predictions to CSV once per market date."""
    last_date = df["TRADE DATE"].max()
    if 'last_save' not in st.session_state or st.session_state.last_save != last_date:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        file_path = OUTPUT_DIR / f"ali_tahmin_{ts}.csv"
        
        export_df = df[['ticker', 'TRADE DATE', 'CLOSING PRICE', 'Target', 'prob_up']].copy()
        export_df.columns = ['Hisse', 'Tarih', 'Fiyat', 'Hedef_Fiyat', 'Olasilik']
        export_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        
        st.session_state.last_save = last_date
        return file_path
    return None

def run_analysis():
    df = get_market_data()
    features = build_features(df)
    
    # Model Loading
    bundle = joblib.load(Path("models/lgbm.pkl"))
    model = bundle["model"].get("final") if isinstance(bundle["model"], dict) else bundle["model"]
    
    latest_date = features["TRADE DATE"].max()
    today = features[features["TRADE DATE"] == latest_date].copy()
    
    # Feature synchronization
    for col in bundle["feature_cols"]:
        if col not in today.columns:
            today[col] = 0.0
        today[col] = today[col].fillna(0.0)

    # ML Inference
    today["prob_up"] = model.predict_proba(today[bundle["feature_cols"]])[:, 1]
    today["Target"] = today["CLOSING PRICE"] * (1 + (today["prob_up"] - 0.5) * 0.12)
    
    def get_signal(p):
        if p >= 0.56: return "🟢 STRONG"
        if p >= 0.52: return "🟡 WATCH"
        return "⚪ NEUTRAL"
    
    today["Signal"] = today["prob_up"].apply(get_signal)
    
    pA = apply_policy_A(today, top_k=10)
    pB = apply_policy_B(today, top_k=10)
    log_file = save_snapshot(today)
    
    return pA, pB, today.sort_values("prob_up", ascending=False), latest_date, log_file

# --- UI Layout ---
st.title("🏛️ Ali Quant Intelligence Terminal")
pA, pB, all_results, lDate, log_path = run_analysis()

# Header Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Analiz Tarihi", lDate.strftime('%d.%m.%Y'))
m2.metric("Kapsam", "BIST 100")
if log_path: st.sidebar.info(f"📁 Kayıt: {log_path.name}")

st.divider()

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("📊 Top 10 Seçki (Policy A)")
    st.dataframe(pA[['ticker', 'Signal', 'CLOSING PRICE', 'Target', 'prob_up']].style.format(
        {'CLOSING PRICE': '{:.2f}', 'Target': '{:.2f}', 'prob_up': '{:.1%}'}), 
        hide_index=True, use_container_width=True)

with col_r:
    st.subheader("🎯 Keskin Nişancı (Policy B)")
    if pB is None or pB.empty:
        st.warning("Bu hafta %56 güven barajını geçen hisse bulunamadı.")
    else:
        st.dataframe(pB[['ticker', 'Signal', 'CLOSING PRICE', 'Target', 'prob_up']].style.format(
            {'CLOSING PRICE': '{:.2f}', 'Target': '{:.2f}', 'prob_up': '{:.1%}'}), 
            hide_index=True, use_container_width=True)

st.divider()

# Full Market Explorer
st.subheader("🌐 Tüm BIST 100 Taraması")
ticker_search = st.text_input("Hisse Ara...", "").upper()
full_list = all_results[['ticker', 'Signal', 'CLOSING PRICE', 'Target', 'prob_up']]
if ticker_search: 
    full_list = full_list[full_list['ticker'].str.contains(ticker_search)]

st.dataframe(full_list.style.format(
    {'CLOSING PRICE': '{:.2f}', 'Target': '{:.2f}', 'prob_up': '{:.1%}'}),
    hide_index=True, use_container_width=True, height=500)

# Auto-refresh
time.sleep(900)
st.rerun()