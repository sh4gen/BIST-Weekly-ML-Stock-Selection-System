# src/scoring.py
from __future__ import annotations
import pandas as pd

def apply_policy_A(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    # Policy A: Standart. Her hafta en yüksek skorlu top_k (5) hisseyi alır.
    s = scored.copy()
    return s.sort_values("prob_up", ascending=False).head(top_k)

def apply_policy_B(scored: pd.DataFrame, top_k: int, candidate_pool: int = 15,
                   close_gt_ema20=False, ret10_gt_0=False, vol_z_gt_0=False, close_gt_vwap=False) -> pd.DataFrame:
    # Policy B: KESKİN NİŞANCI (High Conviction)
    # Sadece kazanma ihtimali %56'nın üzerinde olan hisseleri filtrele
    s = scored.copy()
    s = s[s["prob_up"] >= 0.56]
    
    # Kalanlar arasından en iyilerini seç
    return s.sort_values("prob_up", ascending=False).head(top_k)