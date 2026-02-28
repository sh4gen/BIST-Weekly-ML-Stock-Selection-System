# run_live.py
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from src.features import build_features
from src.scoring import apply_policy_A, apply_policy_B

def fetch_live_bist_data():
    print("1. Yahoo Finance API'sine bağlanılıyor...")
    # Senin paylaştığın güncel BIST 100 listesi
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
    print(f"2. {len(tickers)} adet BIST 100 verisi canlı çekiliyor...")
    # Teknik göstergeler için 6 aylık veri çekiyoruz
    data = yf.download(tickers, period="6mo", group_by="ticker", progress=False)
    
    rows = []
    for ticker in tickers:
        try:
            temp = data[ticker].copy()
            if temp.empty or temp["Close"].isna().all(): continue
            temp = temp.reset_index()
            temp["ticker"] = ticker.replace(".IS", "")
            rows.append(temp)
        except KeyError: continue
            
    all_df = pd.concat(rows, ignore_index=True)
    all_df.rename(columns={"Date": "TRADE DATE", "Close": "CLOSING PRICE", "Open": "OPEN", "High": "HIGH", "Low": "LOW", "Volume": "VOLUME"}, inplace=True)
    
    for c in ["OPEN", "HIGH", "LOW"]:
        if f"{c} PRICE" not in all_df.columns: all_df[f"{c} PRICE"] = all_df[c]
        
    all_df["TRADE DATE"] = pd.to_datetime(all_df["TRADE DATE"]).dt.tz_localize(None)
    return all_df.dropna(subset=["CLOSING PRICE"])

def main():
    print("\n" + "="*60)
    print("🚀 QUANT FONU: BIST 100 CANLI MOTOR BAŞLATILIYOR 🚀")
    print("="*60)

    BASE = Path(__file__).resolve().parent
    all_df = fetch_live_bist_data()
    latest_date = all_df["TRADE DATE"].max()
    print(f"-> Piyasa Tarihi: {latest_date.date()}")

    print("3. Yapay zeka göstergeleri hesaplanıyor...")
    feat = build_features(all_df)
    
    print("4. Model dosyası analiz ediliyor...")
    model_path = BASE / "models" / "lgbm.pkl"
    bundle = joblib.load(model_path)
    model_dict = bundle["model"]
    feature_cols = bundle["feature_cols"]
    
    if isinstance(model_dict, dict):
        active_model = model_dict.get("final", list(model_dict.values())[0])
    else:
        active_model = model_dict

    print(f"-> AKTİF MOTOR: {type(active_model).__name__}")

    today_data = feat[feat["TRADE DATE"] == latest_date].copy()
    for col in feature_cols:
        if col not in today_data.columns: today_data[col] = 0.0
        today_data[col] = today_data[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("5. Tahminler ve Hedef Fiyatlar hesaplanıyor...\n")
    today_data["prob_up"] = active_model.predict_proba(today_data[feature_cols])[:, 1]
    
    # BIST 100 daha büyük olduğu için ilk 10 hisseye bakıyoruz
    picks_A = apply_policy_A(today_data, top_k=10)
    picks_B = apply_policy_B(today_data, top_k=10)

    print("📊" + "-"*25 + " SONUÇLAR " + "-"*25 + "📊")
    
    def print_results(picks, title):
        print(f"\n[{title}]")
        if picks is None or picks.empty:
            print(" 🚨 DİKKAT: Uygun hisse bulunamadı veya baraj geçilemedi!")
            return
            
        for i, row in enumerate(picks.itertuples(), 1):
            # Fiyatı güvenli çekme
            curr = getattr(row, 'CLOSING_PRICE', None)
            if curr is None:
                # Sütun sırasına göre fallback (Genelde ticker'dan hemen sonra fiyattır)
                curr = row[3] 
            
            # Beklenen getiri simülasyonu
            expected_return = (row.prob_up - 0.50) * 0.15 
            target = curr * (1 + expected_return)
            
            print(f" {i:2}. {row.ticker:6} | Mevcut: {curr:8.2f} | Hedef: ~{target:8.2f} | İhtimal: %{row.prob_up*100:.2f}")

    print_results(picks_A, "POLICY A: SÜREKLİ OYUNCU (En İyi 10)")
    print_results(picks_B, "POLICY B: KESKİN NİŞANCI (%56 Barajı)")
            
    print("\n" + "="*60)
    print("Bol kazançlar Ali Patron!")

if __name__ == "__main__":
    main()