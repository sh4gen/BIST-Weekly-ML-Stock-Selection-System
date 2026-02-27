import pandas as pd
import numpy as np

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def zscore_roll(s: pd.Series, window=20) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / (sd + 1e-9)

def build_features(all_df: pd.DataFrame) -> pd.DataFrame:
    df = all_df.sort_values(["ticker", "TRADE DATE"]).copy()

    out = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("TRADE DATE").copy()
        c = g["CLOSING PRICE"]

        g["ret_5"] = c.pct_change(5)
        g["ret_10"] = c.pct_change(10)
        g["ret_20"] = c.pct_change(20)

        g["ema20"] = ema(c, 20)
        g["ema50"] = ema(c, 50)
        g["close_over_ema20"] = (c / (g["ema20"] + 1e-9)) - 1.0
        g["ema20_over_ema50"] = (g["ema20"] / (g["ema50"] + 1e-9)) - 1.0

        g["rsi14"] = rsi(c, 14)

        # VWAP deviation
        if "VWAP" in g.columns:
            g["close_over_vwap"] = (c / (g["VWAP"] + 1e-9)) - 1.0

        # Volume z
        if "TOTAL TRADED VOLUME" in g.columns:
            g["vol_z20"] = zscore_roll(g["TOTAL TRADED VOLUME"], 20)

        # Opening session ratio
        if "TRADED VOLUME AT OPENING SESSION" in g.columns and "TOTAL TRADED VOLUME" in g.columns:
            g["open_vol_ratio"] = g["TRADED VOLUME AT OPENING SESSION"] / (g["TOTAL TRADED VOLUME"] + 1e-9)

        # Short sale ratio
        if "TRADED VALUE OF SHORT SALE TRADES" in g.columns and "TOTAL TRADED VALUE" in g.columns:
            g["short_value_ratio"] = g["TRADED VALUE OF SHORT SALE TRADES"] / (g["TOTAL TRADED VALUE"] + 1e-9)

        out.append(g)

    feat = pd.concat(out, ignore_index=True)

    # seçilecek feature kolonları
    feature_cols = [
        "ret_5","ret_10","ret_20",
        "close_over_ema20","ema20_over_ema50",
        "rsi14",
        "close_over_vwap",
        "vol_z20",
        "open_vol_ratio",
        "short_value_ratio",
    ]
    keep = ["TRADE DATE","ticker","CLOSING PRICE"] + [c for c in feature_cols if c in feat.columns]
    return feat[keep].copy()
