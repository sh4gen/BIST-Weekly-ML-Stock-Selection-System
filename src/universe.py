# src/universe.py
import pandas as pd
import numpy as np

# Tüm backtest boyunca hesaplamaları bir kere yapıp hafızada tutacak küresel önbellek
_UNIVERSE_CACHE = {}

def select_universe_top_liq_vol(
    all_df: pd.DataFrame,
    asof_date: pd.Timestamp,
    liquidity_lookback=60,
    liquidity_top_n=80,
    vol_lookback=20,
    vol_top_n=50
) -> list[str]:
    global _UNIVERSE_CACHE
    
    # Veriler ilk kez istendiğinde MATRİS oluştur ve hafızaya al (Hız %5000 artar)
    if "val_pivot" not in _UNIVERSE_CACHE:
        _UNIVERSE_CACHE["val_pivot"] = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="TOTAL TRADED VALUE").sort_index()
        
        # ATR (Volatilite) matrisi hazırlığı
        high = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="HIGHEST PRICE").sort_index()
        low = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="LOWEST PRICE").sort_index()
        close = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="CLOSING PRICE").sort_index()
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.DataFrame(np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)), index=close.index, columns=close.columns)
        atr_pct = tr.rolling(vol_lookback).mean() / close
        
        _UNIVERSE_CACHE["atr_pct"] = atr_pct

    val_pivot = _UNIVERSE_CACHE["val_pivot"]
    atr_pct = _UNIVERSE_CACHE["atr_pct"]

    # 1. Likidite Filtresi (Anında matris üzerinden çekilir)
    past_vals = val_pivot.loc[:asof_date]
    if len(past_vals) < 10:
        return []
        
    recent_vals = past_vals.tail(liquidity_lookback)
    avg_val = recent_vals.mean()
    valid_counts = recent_vals.notna().sum()
    
    # Yeterli verisi olanları filtrele
    avg_val = avg_val[valid_counts >= max(10, liquidity_lookback // 2)]
    top_liq = avg_val.sort_values(ascending=False).head(liquidity_top_n).index.tolist()
    
    if not top_liq:
        return []

    # 2. Volatilite (ATR) Filtresi
    past_atr = atr_pct.loc[:asof_date, top_liq]
    if past_atr.empty:
        return top_liq[:vol_top_n]
        
    current_atr = past_atr.iloc[-1].dropna()
    top_vol = current_atr.sort_values(ascending=False).head(vol_top_n).index.tolist()

    return top_vol