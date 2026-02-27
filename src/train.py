import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score

def walk_forward_splits(df: pd.DataFrame):
    # 2015-2018 train -> 2019 test, 2015-2019 -> 2020, 2015-2020 -> 2021
    years = sorted(df["TRADE DATE"].dt.year.unique())
    for test_year in [2019, 2020, 2021]:
        if test_year not in years:
            continue
        train = df[df["TRADE DATE"].dt.year < test_year]
        test = df[df["TRADE DATE"].dt.year == test_year]
        yield test_year, train, test

def train_model(df: pd.DataFrame, feature_cols: list[str]):
    df = df.dropna(subset=feature_cols + ["y_up"]).copy()
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    reports = []
    for test_year, train, test in walk_forward_splits(df):
        model.fit(train[feature_cols], train["y_up"])
        pred = model.predict(test[feature_cols])
        p = precision_score(test["y_up"], pred, zero_division=0)
        reports.append({"test_year": test_year, "precision": float(p), "n_test": int(len(test))})

    # final fit on all
    model.fit(df[feature_cols], df["y_up"])
    return model, pd.DataFrame(reports)
