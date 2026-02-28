import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score

def train_model(df: pd.DataFrame, feature_cols: list[str]):
    # YENİ EKLENEN SATIR: XGBoost'u çökerten 'Sonsuzluk' (inf) değerlerini NaN'a çeviriyoruz
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Ve şimdi o hatalı satırları güvenle siliyoruz
    df = df.dropna(subset=feature_cols + ["y_up"]).copy()
    
    models = {}
    reports = []
    years = sorted(df["TRADE DATE"].dt.year.unique())
    
    # YENİ MOTOR: XGBoost (Agresif ve Keskin)
    model_params = {
        "n_estimators": 150, 
        "learning_rate": 0.05, 
        "max_depth": 4, 
        "subsample": 0.7, 
        "colsample_bytree": 0.7, 
        "reg_alpha": 0.5,    
        "reg_lambda": 0.5,   
        "random_state": 42,
        "n_jobs": -1
    }
    
    for target_year in years:
        train = df[df["TRADE DATE"].dt.year < target_year]
        test = df[df["TRADE DATE"].dt.year == target_year]
        
        if train.empty:
            continue
            
        model = XGBClassifier(**model_params)
        model.fit(train[feature_cols], train["y_up"])
        models[target_year] = model
        
        if not test.empty:
            pred = model.predict(test[feature_cols])
            p = precision_score(test["y_up"], pred, zero_division=0)
            reports.append({"test_year": target_year, "precision": float(p), "n_test": int(len(test))})

    final_model = XGBClassifier(**model_params)
    final_model.fit(df[feature_cols], df["y_up"])
    models["final"] = final_model

    print("\n--- XGBOOST FEATURE IMPORTANCE ---")
    importances = final_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    print(imp_df.to_string())
    print("----------------------------------\n")

    return models, pd.DataFrame(reports)