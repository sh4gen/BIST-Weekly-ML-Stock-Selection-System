# run_backtest.py
from __future__ import annotations
import yaml
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_layer import load_equities_folder
from src.features import build_features
from src.labels import add_labels
from src.universe import select_universe_top_liq_vol
from src.scoring import apply_policy_A, apply_policy_B
from src.backtest import run_weekly_backtest

def main():
    BASE = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(BASE / "config" / "settings.yaml", "r", encoding="utf-8"))

    all_df = load_equities_folder(
        equities_dir=cfg["paths"]["equities_dir"],
        test_csv_path=cfg["paths"].get("test_csv")
    )
    all_df["TRADE DATE"] = pd.to_datetime(all_df["TRADE DATE"])
    print("Equities date range:", all_df["TRADE DATE"].min().date(), "->", all_df["TRADE DATE"].max().date())

    feat = build_features(all_df)
    tcfg = cfg["target"]
    ds = add_labels(
        feat,
        horizon_days=tcfg["horizon_days"],
        up_threshold=tcfg["up_threshold"],
        risk_horizon_days=tcfg["risk_horizon_days"],
        max_drawdown_threshold=tcfg["max_drawdown_threshold"],
    )

    bundle = joblib.load(BASE / "models" / "lgbm.pkl")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    ucfg = cfg["universe"]
    def universe_fn(all_df_, asof_date):
        return select_universe_top_liq_vol(
            all_df_,
            asof_date=asof_date,
            liquidity_lookback=ucfg["liquidity_lookback"],
            liquidity_top_n=ucfg["liquidity_top_n"],
            vol_lookback=ucfg["vol_lookback"],
            vol_top_n=ucfg["vol_top_n"],
        )

    scfg = cfg["selection"]
    confirm = scfg.get("confirm", {})

    res = run_weekly_backtest(
        ds=ds,
        all_df=all_df,
        model=model,
        feature_cols=feature_cols,
        select_universe_fn=universe_fn,
        apply_A_fn=apply_policy_A,
        apply_B_fn=apply_policy_B,
        holding_days=5,
        top_k=scfg["top_k"],
        candidate_pool=scfg.get("candidate_pool", 15),
        confirm=confirm,
        start_date="2016-01-01",
        end_date=str(all_df["TRADE DATE"].max().date()),
    )

    outdir = BASE / "reports"
    outdir.mkdir(exist_ok=True)

    res.equity.to_csv(outdir / "backtest_equity_curve.csv", index=False)
    res.trades_A.to_csv(outdir / "backtest_trades_A.csv", index=False)
    res.trades_B.to_csv(outdir / "backtest_trades_B.csv", index=False)
    res.summary.to_csv(outdir / "backtest_summary.csv", index=False)

    print("\nBacktest Summary:")
    print(res.summary)

    # plot
    if not res.equity.empty:
        plt.figure(figsize=(10,5), dpi=150)
        plt.plot(res.equity["entry_date"], res.equity["equity_A"], label="Policy A")
        plt.plot(res.equity["entry_date"], res.equity["equity_B"], label="Policy B")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.title("Weekly Backtest Equity Curves (Hold=5 trading days)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "backtest_equity_curve.png", bbox_inches="tight")
        plt.close()

        print("Saved plot:", outdir / "backtest_equity_curve.png")

if __name__ == "__main__":
    main()
