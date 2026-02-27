from __future__ import annotations
import pandas as pd
from pathlib import Path

# attributes.txt'e göre alanlar:
# TRADE DATE, OPENING PRICE, HIGHEST PRICE, LOWEST PRICE, CLOSING PRICE,
# VWAP, TOTAL TRADED VALUE, TOTAL TRADED VOLUME, ... :contentReference[oaicite:0]{index=0}

NUM_COLS = [
    "OPENING PRICE", "HIGHEST PRICE", "LOWEST PRICE", "CLOSING PRICE",
    "VWAP",
    "TOTAL TRADED VALUE", "TOTAL TRADED VOLUME",
    "TRADED VALUE AT OPENING SESSION", "TRADED VOLUME AT OPENING SESSION",
    "TRADED VALUE OF SHORT SALE TRADES", "TRADED VOLUME OF SHORT SALE TRADES",
    "SHORT SALE VWAP",
    "CHANGE TO PREVIOUS CLOSING (%)",
]

def load_equity_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tarih
    df["TRADE DATE"] = pd.to_datetime(df["TRADE DATE"], errors="coerce")
    # numerikler
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TRADE DATE"]).sort_values("TRADE DATE")
    # ticker
    ticker = Path(path).stem.upper()
    df["ticker"] = ticker
    return df

def load_equities_folder(equities_dir: str | Path) -> pd.DataFrame:
    equities_dir = Path(equities_dir)
    files = sorted(equities_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in: {equities_dir}")
    frames = [load_equity_csv(p) for p in files]
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def load_xu100_from_price_indices(csv_path: str | Path) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    from pathlib import Path

    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # zorunlu kolonlar
    for col in ["ENDEKS KODU", "TARIH", "KAPANIS"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}. Columns={list(df.columns)}")

    # normalize
    df["ENDEKS KODU"] = df["ENDEKS KODU"].astype(str).str.upper().str.strip()
    df["TARIH"] = df["TARIH"].astype(str).str.strip()
    df["KAPANIS"] = df["KAPANIS"].astype(str).str.strip()

    # header-like junk satırları ele (senin listede "INDEX CODE" var)
    df = df[~df["ENDEKS KODU"].isin(["INDEX CODE", "ENDEKS KODU"])].copy()

    # --- Tarih: birden fazla format dene (uyarı yok) ---
    s = df["TARIH"]

    t = pd.Series(pd.NaT, index=df.index)
    fmts = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
    for fmt in fmts:
        mask = t.isna()
        if not mask.any():
            break
        t.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")

    # son çare: dayfirst parse (format vermeden) — burada uyarı istemiyorsan
    # sadece hala NaT kalan az satıra uygulanıyor
    mask = t.isna()
    if mask.any():
        t.loc[mask] = pd.to_datetime(s[mask], dayfirst=True, errors="coerce")

    df["TARIH"] = t

    # --- KAPANIS: güvenli numeric parse ---
    # sadece virgülü noktaya çeviriyoruz, NOKTA SİLMİYORUZ (önceki bug burada olabiliyordu)
    c = df["KAPANIS"].str.replace(" ", "", regex=False)
    # eğer virgül varsa, decimal comma olabilir -> '.' dokunma, sadece ',' -> '.'
    c = c.str.replace(",", ".", regex=False)
    df["KAPANIS"] = pd.to_numeric(c, errors="coerce")

    # temizle
    df = df.dropna(subset=["TARIH", "KAPANIS"]).copy()

    # --- XU100 seç (önce exact, yoksa startswith) ---
    xu = df[df["ENDEKS KODU"] == "XU100"].copy()
    if xu.empty:
        xu = df[df["ENDEKS KODU"].str.startswith("XU100", na=False)].copy()

    if xu.empty:
        return pd.DataFrame(columns=["date", "close"])

    xu = xu.sort_values("TARIH")[["TARIH", "KAPANIS"]]
    xu = xu.rename(columns={"TARIH": "date", "KAPANIS": "close"})
    return xu