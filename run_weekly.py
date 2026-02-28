# run_weekly.py
from __future__ import annotations
import yaml
from pathlib import Path
import pandas as pd
import joblib

from src.data_layer import load_equities_folder, load_xu100_from_price_indices
from src.features import build_features
from src.labels import add_labels
from src.universe import select_universe_top_liq_vol
from src.scoring import apply_policy_A, apply_policy_B
from src.report import build_weekly_pdf

def main():
    BASE = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(BASE / "config" / "settings.yaml", "r", encoding="utf-8"))

    # 1) veriyi yükle
# Yeni Satır:
    all_df = load_equities_folder(
        equities_dir=cfg["paths"]["equities_dir"],
        test_csv_path=cfg["paths"].get("test_csv")
    )
    all_df["TRADE DATE"] = pd.to_datetime(all_df["TRADE DATE"])
    last_date = all_df["TRADE DATE"].max()
    print("Last trade date in equities:", last_date.date())

    # 2) universe seç (as-of last date)
    ucfg = cfg["universe"]
    universe = select_universe_top_liq_vol(
        all_df,
        asof_date=last_date,
        liquidity_lookback=ucfg["liquidity_lookback"],
        liquidity_top_n=ucfg["liquidity_top_n"],
        vol_lookback=ucfg["vol_lookback"],
        vol_top_n=ucfg["vol_top_n"],
    )
    print(f"Universe size: {len(universe)}")

    # 3) feature üret + label
    feat = build_features(all_df)
    tcfg = cfg["target"]
    ds = add_labels(
        feat,
        horizon_days=tcfg["horizon_days"],
        up_threshold=tcfg["up_threshold"],
        risk_horizon_days=tcfg["risk_horizon_days"],
        max_drawdown_threshold=tcfg["max_drawdown_threshold"],
    )

    # 4) sadece son gün + universe filtre
    today = ds[ds["TRADE DATE"] == last_date].copy()
    today = today[today["ticker"].isin(universe)].copy()

    # modelde kullanılan feature kolonlarıyla hizala
    bundle = joblib.load(BASE / "models" / "lgbm.pkl")
    models_dict = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Canlı sistem olduğu için en son eğitilen 'final' modelini al
    model = models_dict["final"]

    # NaN temizle
    today = today.dropna(subset=feature_cols).copy()
    if today.empty:
        print("No rows to score after filtering (today is empty).")
        return

    # 5) skorla
    today["prob_up"] = model.predict_proba(today[feature_cols])[:, 1]

    # 6) policy A/B
    scfg = cfg["selection"]
    top_k = scfg["top_k"]

    picks_A = apply_policy_A(today, top_k=top_k)

    conf = scfg.get("confirm", {})
    picks_B = apply_policy_B(
        today,
        top_k=top_k,
        candidate_pool=scfg.get("candidate_pool", 15),
        close_gt_ema20=conf.get("close_gt_ema20", True),
        ret10_gt_0=conf.get("ret10_gt_0", True),
        vol_z_gt_0=conf.get("vol_z_gt_0", True),
        close_gt_vwap=conf.get("close_gt_vwap", False),
    )

    # overlap
    overlap = sorted(set(picks_A["ticker"]).intersection(set(picks_B["ticker"])))
    overlap_df = today[today["ticker"].isin(overlap)].sort_values("prob_up", ascending=False)

    # 7) kaydet
    outdir = BASE / "reports"
    outdir.mkdir(exist_ok=True)

    stamp = str(last_date.date())
    cols_show = ["ticker", "prob_up", "risk_flag", "ret_5", "ret_10", "ret_20",
                 "close_over_ema20", "ema20_over_ema50", "rsi14"]
    cols_show = [c for c in cols_show if c in today.columns]

    picks_A[cols_show].to_csv(outdir / f"weekly_picks_A_{stamp}.csv", index=False)
    picks_B[cols_show].to_csv(outdir / f"weekly_picks_B_{stamp}.csv", index=False)
    overlap_df[cols_show].to_csv(outdir / f"weekly_overlap_{stamp}.csv", index=False)

    print("Saved:")
    print(" -", outdir / f"weekly_picks_A_{stamp}.csv")
    print(" -", outdir / f"weekly_picks_B_{stamp}.csv")
    print(" -", outdir / f"weekly_overlap_{stamp}.csv")
    print("Overlap tickers:", overlap)

    # 8) PDF
    xu100_df = load_xu100_from_price_indices(cfg["paths"]["index_prices_csv"])

    # hisse datasının tarih aralığına kırp
    min_d = all_df["TRADE DATE"].min()
    max_d = all_df["TRADE DATE"].max()

    xu100_df = xu100_df[(xu100_df["date"] >= min_d) & (xu100_df["date"] <= max_d)].copy()

    if xu100_df is not None and not xu100_df.empty:
        xu100_df = xu100_df[xu100_df["date"] <= last_date].copy()

    pdf_path = outdir / f"weekly_report_{stamp}.pdf"
    build_weekly_pdf(
        out_pdf=pdf_path,
        stamp=stamp,
        all_equities_df=all_df,
        picks_A=picks_A,
        picks_B=picks_B,
        overlap=overlap_df,
        xu100_df=xu100_df,
    )
    print("PDF saved:", pdf_path)

if __name__ == "__main__":
    main()
