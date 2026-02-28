import yaml
from pathlib import Path
import joblib

from src.data_layer import load_equities_folder
from src.features import build_features
from src.labels import add_labels
from src.train import train_model

def main():
    cfg = yaml.safe_load(open("config/settings.yaml", "r", encoding="utf-8"))

    all_df = load_equities_folder(
        equities_dir=cfg["paths"]["equities_dir"],
        test_csv_path=cfg["paths"].get("test_csv")
    )
    feat = build_features(all_df)
    ds = add_labels(
        feat,
        horizon_days=cfg["target"]["horizon_days"],
        up_threshold=cfg["target"]["up_threshold"],
        risk_horizon_days=cfg["target"]["risk_horizon_days"],
        max_drawdown_threshold=cfg["target"]["max_drawdown_threshold"],
    )

    feature_cols = [c for c in ds.columns if c not in ["TRADE DATE","ticker","CLOSING PRICE","fwd_ret","y_up","fwd_maxdd","risk_flag"]]
    model, rep = train_model(ds, feature_cols)

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, "models/lgbm.pkl")
    rep.to_csv("models/walk_forward_report.csv", index=False)
    print(rep)

if __name__ == "__main__":
    main()
