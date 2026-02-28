# src/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades_A: pd.DataFrame
    trades_B: pd.DataFrame
    summary: pd.DataFrame

def _next_trading_day(dates: np.ndarray, d: pd.Timestamp) -> pd.Timestamp | None:
    idx = np.searchsorted(dates, np.datetime64(d), side="left")
    if idx >= len(dates): return None
    return pd.Timestamp(dates[idx])

def _advance_n_days(dates: np.ndarray, d: pd.Timestamp, n: int) -> pd.Timestamp | None:
    idx = np.searchsorted(dates, np.datetime64(d), side="left")
    idx2 = idx + n
    if idx2 >= len(dates): return None
    return pd.Timestamp(dates[idx2])

# YENİ: Ters Volatilite Ağırlıklı Getiri Hesaplayıcı
def _portfolio_return_weighted(prices_entry: pd.Series, prices_exit: pd.Series, weights: dict[str, float]) -> float:
    # 1. Kirli Veri Koruması: Aynı hisse iki kere geldiyse birini sil (Duplicate Drop)
    prices_entry = prices_entry[~prices_entry.index.duplicated(keep='first')]
    prices_exit = prices_exit[~prices_exit.index.duplicated(keep='first')]
    
    common = prices_entry.index.intersection(prices_exit.index)
    if len(common) == 0: return 0.0
    
    rets = (prices_exit[common] / prices_entry[common]) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty: return 0.0
    
    # 2. Vektörel Hızlı Çarpım (For döngüsü kullanmıyoruz, Series üzerinden direkt çarpıyoruz)
    w_series = pd.Series(weights, dtype=float)
    common_w = rets.index.intersection(w_series.index)
    
    if len(common_w) == 0: return 0.0
    
    r = rets[common_w]
    w = w_series[common_w]
    
    total_w = w.sum()
    if total_w == 0: return 0.0
    
    return float((r * w).sum() / total_w)

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def _sharpe(returns: pd.Series, periods_per_year=52) -> float:
    r = returns.dropna()
    if len(r) < 5: return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd): return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))

def run_weekly_backtest(ds, all_df, model, feature_cols, select_universe_fn, apply_A_fn, apply_B_fn, 
                        holding_days=5, top_k=5, candidate_pool=15, confirm=None, 
                        start_date="2016-01-01", end_date="2026-12-31") -> BacktestResult:
    confirm = confirm or {}
    cal = np.array(sorted(all_df["TRADE DATE"].dropna().unique()))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    px_close = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="CLOSING PRICE", aggfunc="last").sort_index()
    ds_indexed = ds.set_index("TRADE DATE").sort_index()

    equity_A, equity_B = 1.0, 1.0
    equity_rows, trades_A, trades_B = [], [], []
    d = start_date

    while True:
        signal_date = _next_trading_day(cal, d)
        if signal_date is None or signal_date > end_date: break

        execution_date = _advance_n_days(cal, signal_date, 1)
        if execution_date is None or execution_date > end_date: break

        exit_date = _advance_n_days(cal, execution_date, holding_days)
        if exit_date is None or exit_date > end_date: break

        if signal_date not in ds_indexed.index:
            d = signal_date + pd.Timedelta(days=7)
            continue
            
        today = ds_indexed.loc[[signal_date]].copy()
        universe = select_universe_fn(all_df, asof_date=signal_date)
        
        today = today[today["ticker"].isin(universe)].copy()
        today = today.dropna(subset=feature_cols).copy()

        if today.empty:
            d = signal_date + pd.Timedelta(days=7)
            continue

        market_is_crashing = False
        if "ret_20" in today.columns:
            market_health = today["ret_20"].median()
            if market_health < -0.02: 
                market_is_crashing = True

        if isinstance(model, dict):
            target_year = signal_date.year
            valid_years = [y for y in model.keys() if isinstance(y, (int, np.integer)) and y <= target_year]
            if not valid_years:
                d = signal_date + pd.Timedelta(days=7)
                continue
            active_model = model[max(valid_years)]
        else:
            active_model = model

        today["prob_up"] = active_model.predict_proba(today[feature_cols])[:, 1]
        
        if market_is_crashing:
            picks_A = pd.DataFrame()
            picks_B = pd.DataFrame()
        else:
            picks_A = apply_A_fn(today, top_k=top_k)
            picks_B = apply_B_fn(today, top_k=top_k, candidate_pool=candidate_pool, 
                                 close_gt_ema20=confirm.get("close_gt_ema20", True), 
                                 ret10_gt_0=confirm.get("ret10_gt_0", True), 
                                 vol_z_gt_0=confirm.get("vol_z_gt_0", True), 
                                 close_gt_vwap=confirm.get("close_gt_vwap", False))

        tickers_A = picks_A["ticker"].tolist() if not picks_A.empty else []
        tickers_B = picks_B["ticker"].tolist() if not picks_B.empty else []

        # YENİ: Ağırlıkların Hesaplanması (Çok Oynak Olana Az Para, Az Oynak Olana Çok Para)
        w_A, w_B = {}, {}
        
        if tickers_A:
            atr_A = today[today["ticker"].isin(tickers_A)].set_index("ticker")["atr_pct"]
            inv_vol_A = 1.0 / (atr_A + 1e-6) # 0'a bölünme hatasını engeller
            w_A = (inv_vol_A / inv_vol_A.sum()).to_dict()
            
        if tickers_B:
            atr_B = today[today["ticker"].isin(tickers_B)].set_index("ticker")["atr_pct"]
            inv_vol_B = 1.0 / (atr_B + 1e-6)
            w_B = (inv_vol_B / inv_vol_B.sum()).to_dict()

        if execution_date in px_close.index and exit_date in px_close.index:
            rA = _portfolio_return_weighted(px_close.loc[execution_date, tickers_A], px_close.loc[exit_date, tickers_A], w_A) if tickers_A else 0.0
            rB = _portfolio_return_weighted(px_close.loc[execution_date, tickers_B], px_close.loc[exit_date, tickers_B], w_B) if tickers_B else 0.0
        else:
            rA, rB = 0.0, 0.0

        equity_A *= (1.0 + rA)
        equity_B *= (1.0 + rB)

        equity_rows.append({"entry_date": execution_date, "exit_date": exit_date, "ret_A": rA, "ret_B": rB, "equity_A": equity_A, "equity_B": equity_B, "n_A": len(tickers_A), "n_B": len(tickers_B)})
        if tickers_A: trades_A.append({"entry_date": execution_date, "exit_date": exit_date, "tickers": ",".join(tickers_A), "ret": rA, "equity": equity_A})
        if tickers_B: trades_B.append({"entry_date": execution_date, "exit_date": exit_date, "tickers": ",".join(tickers_B), "ret": rB, "equity": equity_B})

        d = signal_date + pd.Timedelta(days=7)

    equity_df = pd.DataFrame(equity_rows)
    trades_A_df = pd.DataFrame(trades_A)
    trades_B_df = pd.DataFrame(trades_B)

    if not equity_df.empty:
        active_weeks_A = equity_df[equity_df["n_A"] > 0]
        active_weeks_B = equity_df[equity_df["n_B"] > 0]
        
        win_A = float((active_weeks_A["ret_A"] > 0).mean()) if not active_weeks_A.empty else 0.0
        win_B = float((active_weeks_B["ret_B"] > 0).mean()) if not active_weeks_B.empty else 0.0
        
        mdd_A, mdd_B = _max_drawdown(equity_df["equity_A"]), _max_drawdown(equity_df["equity_B"])
        sharpe_A, sharpe_B = _sharpe(equity_df["ret_A"]), _sharpe(equity_df["ret_B"])
        total_A, total_B = float(equity_df["equity_A"].iloc[-1] - 1.0), float(equity_df["equity_B"].iloc[-1] - 1.0)
        avg_A = float(active_weeks_A["ret_A"].mean()) if not active_weeks_A.empty else 0.0
        avg_B = float(active_weeks_B["ret_B"].mean()) if not active_weeks_B.empty else 0.0
        ntr = len(active_weeks_A)
    else:
        win_A = win_B = mdd_A = mdd_B = sharpe_A = sharpe_B = total_A = total_B = avg_A = avg_B = float("nan")
        ntr = 0

    summary = pd.DataFrame([
        {"strategy": "A", "trades": ntr, "total_return": total_A, "avg_weekly_return": avg_A, "win_rate": win_A, "max_drawdown": mdd_A, "sharpe": sharpe_A},
        {"strategy": "B", "trades": ntr, "total_return": total_B, "avg_weekly_return": avg_B, "win_rate": win_B, "max_drawdown": mdd_B, "sharpe": sharpe_B},
    ])
    return BacktestResult(equity=equity_df, trades_A=trades_A_df, trades_B=trades_B_df, summary=summary)