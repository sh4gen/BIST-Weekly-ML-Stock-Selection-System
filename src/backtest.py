# src/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.DataFrame          # date, equity_A, equity_B
    trades_A: pd.DataFrame        # trade logs
    trades_B: pd.DataFrame
    summary: pd.DataFrame         # metrics table

def _next_trading_day(dates: np.ndarray, d: pd.Timestamp) -> pd.Timestamp | None:
    idx = np.searchsorted(dates, np.datetime64(d), side="left")
    if idx >= len(dates):
        return None
    return pd.Timestamp(dates[idx])

def _advance_n_days(dates: np.ndarray, d: pd.Timestamp, n: int) -> pd.Timestamp | None:
    idx = np.searchsorted(dates, np.datetime64(d), side="left")
    idx2 = idx + n
    if idx2 >= len(dates):
        return None
    return pd.Timestamp(dates[idx2])

def _portfolio_return_equal_weight(prices_entry: pd.Series, prices_exit: pd.Series) -> float:
    # equal weight across tickers available both entry & exit
    common = prices_entry.index.intersection(prices_exit.index)
    if len(common) == 0:
        return 0.0
    rets = (prices_exit[common] / prices_entry[common]) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return 0.0
    return float(rets.mean())

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def _sharpe(returns: pd.Series, periods_per_year=52) -> float:
    r = returns.dropna()
    if len(r) < 5:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))

def run_weekly_backtest(
    ds: pd.DataFrame,
    all_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    select_universe_fn,
    apply_A_fn,
    apply_B_fn,
    holding_days: int = 5,
    top_k: int = 5,
    candidate_pool: int = 15,
    confirm: dict | None = None,
    start_date: str | pd.Timestamp = "2016-01-01",
    end_date: str | pd.Timestamp = "2021-12-31",
) -> BacktestResult:
    """
    ds: features+labels table (TRADE DATE, ticker, CLOSING PRICE, ...features..., risk_flag)
    all_df: raw equities table (TRADE DATE, ticker, CLOSING PRICE ...)
    """

    confirm = confirm or {}

    # trading calendar from equities (unique dates)
    cal = np.array(sorted(all_df["TRADE DATE"].dropna().unique()))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # price pivot for fast lookup (close prices)
    px = all_df.pivot_table(index="TRADE DATE", columns="ticker", values="CLOSING PRICE", aggfunc="last").sort_index()

    equity_A = 1.0
    equity_B = 1.0

    equity_rows = []
    trades_A = []
    trades_B = []

    # iterate weekly: every 7 calendar days, but snap to next trading day
    d = start_date
    while True:
        entry_date = _next_trading_day(cal, d)
        if entry_date is None or entry_date > end_date:
            break

        exit_date = _advance_n_days(cal, entry_date, holding_days)
        if exit_date is None or exit_date > end_date:
            break

        # universe selection as-of entry_date
        universe = select_universe_fn(all_df, asof_date=entry_date)

        # build "today" snapshot from ds at entry_date
        today = ds[ds["TRADE DATE"] == entry_date].copy()
        today = today[today["ticker"].isin(universe)].copy()
        today = today.dropna(subset=feature_cols).copy()

        if today.empty:
            # move forward one week
            d = entry_date + pd.Timedelta(days=7)
            continue

        probs = model.predict_proba(today[feature_cols])[:, 1]
        today["prob_up"] = probs

        # pick A/B
        picks_A = apply_A_fn(today, top_k=top_k)
        picks_B = apply_B_fn(
            today,
            top_k=top_k,
            candidate_pool=candidate_pool,
            close_gt_ema20=confirm.get("close_gt_ema20", True),
            ret10_gt_0=confirm.get("ret10_gt_0", True),
            vol_z_gt_0=confirm.get("vol_z_gt_0", True),
            close_gt_vwap=confirm.get("close_gt_vwap", False),
        )

        tickers_A = picks_A["ticker"].tolist()
        tickers_B = picks_B["ticker"].tolist()

        # compute realized equal-weight returns over holding window
        if entry_date in px.index and exit_date in px.index:
            rA = _portfolio_return_equal_weight(px.loc[entry_date, tickers_A], px.loc[exit_date, tickers_A]) if tickers_A else 0.0
            rB = _portfolio_return_equal_weight(px.loc[entry_date, tickers_B], px.loc[exit_date, tickers_B]) if tickers_B else 0.0
        else:
            rA = 0.0
            rB = 0.0

        equity_A *= (1.0 + rA)
        equity_B *= (1.0 + rB)

        equity_rows.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "ret_A": rA,
            "ret_B": rB,
            "equity_A": equity_A,
            "equity_B": equity_B,
            "n_A": len(tickers_A),
            "n_B": len(tickers_B),
        })

        trades_A.append({
            "entry_date": entry_date, "exit_date": exit_date,
            "tickers": ",".join(tickers_A),
            "ret": rA, "equity": equity_A
        })
        trades_B.append({
            "entry_date": entry_date, "exit_date": exit_date,
            "tickers": ",".join(tickers_B),
            "ret": rB, "equity": equity_B
        })

        # next week
        d = entry_date + pd.Timedelta(days=7)

    equity_df = pd.DataFrame(equity_rows)
    trades_A_df = pd.DataFrame(trades_A)
    trades_B_df = pd.DataFrame(trades_B)

    # summary metrics
    if not equity_df.empty:
        win_A = float((equity_df["ret_A"] > 0).mean())
        win_B = float((equity_df["ret_B"] > 0).mean())
        mdd_A = _max_drawdown(equity_df["equity_A"])
        mdd_B = _max_drawdown(equity_df["equity_B"])
        sharpe_A = _sharpe(equity_df["ret_A"])
        sharpe_B = _sharpe(equity_df["ret_B"])
        total_A = float(equity_df["equity_A"].iloc[-1] - 1.0)
        total_B = float(equity_df["equity_B"].iloc[-1] - 1.0)
        avg_A = float(equity_df["ret_A"].mean())
        avg_B = float(equity_df["ret_B"].mean())
        ntr = int(len(equity_df))
    else:
        win_A = win_B = mdd_A = mdd_B = sharpe_A = sharpe_B = total_A = total_B = avg_A = avg_B = float("nan")
        ntr = 0

    summary = pd.DataFrame([
        {"strategy": "A", "trades": ntr, "total_return": total_A, "avg_weekly_return": avg_A, "win_rate": win_A, "max_drawdown": mdd_A, "sharpe": sharpe_A},
        {"strategy": "B", "trades": ntr, "total_return": total_B, "avg_weekly_return": avg_B, "win_rate": win_B, "max_drawdown": mdd_B, "sharpe": sharpe_B},
    ])

    return BacktestResult(
        equity=equity_df,
        trades_A=trades_A_df,
        trades_B=trades_B_df,
        summary=summary
    )
