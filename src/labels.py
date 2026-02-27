import pandas as pd
import numpy as np

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0

def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    arr = close.to_numpy()
    dd = np.full_like(arr, fill_value=np.nan, dtype=float)

    for i in range(len(arr)):
        if i + horizon >= len(arr):
            break

        if np.isnan(arr[i]) or arr[i] <= 0:
            continue

        window = arr[i:i+horizon+1]
        min_p = np.nanmin(window)

        if np.isnan(min_p) or min_p <= 0:
            continue

        dd[i] = (min_p / arr[i]) - 1.0

    return pd.Series(dd, index=close.index)

def add_labels(df_feat: pd.DataFrame, horizon_days=5, up_threshold=0.025,
               risk_horizon_days=10, max_drawdown_threshold=-0.03) -> pd.DataFrame:
    df = df_feat.sort_values(["ticker","TRADE DATE"]).copy()
    out = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("TRADE DATE").copy()
        close = g["CLOSING PRICE"]

        g["fwd_ret"] = forward_return(close, horizon_days)
        g["y_up"] = (g["fwd_ret"] > up_threshold).astype(int)

        g["fwd_maxdd"] = forward_max_drawdown(close, risk_horizon_days)
        g["risk_flag"] = (g["fwd_maxdd"] < max_drawdown_threshold).astype(int)

        out.append(g)
    return pd.concat(out, ignore_index=True)
