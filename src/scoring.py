# src/scoring.py
from __future__ import annotations
import pandas as pd

def apply_policy_A(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    # score-only, risk_flag=1 olanları elemek istersen burada filtre koyabiliriz
    # öneri: risk_flag=1 filtrele
    s = scored[scored["risk_flag"] == 0].copy()
    return s.sort_values("prob_up", ascending=False).head(top_k)

def apply_policy_B(scored: pd.DataFrame, top_k: int, candidate_pool: int,
                   close_gt_ema20=True, ret10_gt_0=True, vol_z_gt_0=True, close_gt_vwap=False) -> pd.DataFrame:
    # önce model skoru ile aday havuzu
    s = scored[scored["risk_flag"] == 0].copy()
    s = s.sort_values("prob_up", ascending=False).head(candidate_pool)

    # confirmation filtreleri (varsa uygula)
    if close_gt_ema20 and "close_over_ema20" in s.columns:
        s = s[s["close_over_ema20"] > 0]

    if ret10_gt_0 and "ret_10" in s.columns:
        s = s[s["ret_10"] > 0]

    if vol_z_gt_0 and "vol_z20" in s.columns:
        s = s[s["vol_z20"] > 0]

    if close_gt_vwap and "close_over_vwap" in s.columns:
        s = s[s["close_over_vwap"] > 0]

    # kalanlardan top_k seç
    return s.sort_values("prob_up", ascending=False).head(top_k)
