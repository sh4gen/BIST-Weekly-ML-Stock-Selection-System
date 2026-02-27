import pandas as pd
import numpy as np

def compute_atr_pct(g: pd.DataFrame, window=20) -> pd.Series:
    g = g.sort_values("TRADE DATE").copy()
    prev_close = g["CLOSING PRICE"].shift(1)
    tr = np.maximum.reduce([
        (g["HIGHEST PRICE"] - g["LOWEST PRICE"]).abs().to_numpy(),
        (g["HIGHEST PRICE"] - prev_close).abs().to_numpy(),
        (g["LOWEST PRICE"] - prev_close).abs().to_numpy(),
    ])
    tr = pd.Series(tr, index=g.index)
    atr = tr.rolling(window).mean()
    return (atr / g["CLOSING PRICE"]).rename("atr_pct")

def select_universe_top_liq_vol(
    all_df: pd.DataFrame,
    asof_date: pd.Timestamp,
    liquidity_lookback=60,
    liquidity_top_n=80,
    vol_lookback=20,
    vol_top_n=50
) -> list[str]:
    df = all_df[all_df["TRADE DATE"] <= asof_date].copy()
    df = df.sort_values(["ticker", "TRADE DATE"])

    # liquidity: avg traded value last N days
    liq_vals = []
    for t, g in df.groupby("ticker"):
        tail = g.tail(liquidity_lookback)
        if tail["TOTAL TRADED VALUE"].notna().sum() < max(10, liquidity_lookback // 2):
            continue
        liq_vals.append((t, tail["TOTAL TRADED VALUE"].mean()))
    liq = pd.DataFrame(liq_vals, columns=["ticker", "avg_value"]).dropna()
    top_liq = liq.sort_values("avg_value", ascending=False).head(liquidity_top_n)["ticker"].tolist()

    # volatility: ATR% last
    atr_vals = []
    for t, g in df[df["ticker"].isin(top_liq)].groupby("ticker"):
        g = g.tail(max(liquidity_lookback, vol_lookback + 5)).copy()
        g.loc[:, "atr_pct"] = compute_atr_pct(g, window=vol_lookback)
        last = g["atr_pct"].iloc[-1]
        if pd.notna(last):
            atr_vals.append((t, float(last)))
    atrs = pd.DataFrame(atr_vals, columns=["ticker", "atr_pct"]).dropna()
    top_vol = atrs.sort_values("atr_pct", ascending=False).head(vol_top_n)["ticker"].tolist()
    return top_vol
