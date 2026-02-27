# src/report.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _plot_stock_panel(df: pd.DataFrame, ticker: str, out_png: Path, meta: dict):
    """
    df: columns: TRADE DATE, OPENING PRICE, HIGHEST PRICE, LOWEST PRICE, CLOSING PRICE, VWAP(optional)
    meta: dict of metrics to print in title area
    """
    df = df.sort_values("TRADE DATE").tail(220).copy()  # son ~1 yıl iş günü
    d = df["TRADE DATE"]

    close = df["CLOSING PRICE"]
    openp = df.get("OPENING PRICE", close)
    high = df.get("HIGHEST PRICE", close)
    low  = df.get("LOWEST PRICE", close)

    # basit candlestick çizimi (matplotlib ile)
    x = np.arange(len(df))
    fig = plt.figure(figsize=(12, 7), dpi=150)
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 1.0, 1.0, 0.8], hspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # wick + body
    for i in range(len(df)):
        ax1.vlines(x[i], low.iloc[i], high.iloc[i], linewidth=0.6)
        o = float(openp.iloc[i]) if pd.notna(openp.iloc[i]) else float(close.iloc[i])
        c = float(close.iloc[i]) if pd.notna(close.iloc[i]) else o
        bottom = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = (float(high.iloc[i]) - float(low.iloc[i])) * 0.05
        ax1.add_patch(plt.Rectangle((x[i] - 0.3, bottom), 0.6, height, fill=False, linewidth=0.6))

    # EMA
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ax1.plot(x, ema20, linewidth=1.0, label="EMA20")
    ax1.plot(x, ema50, linewidth=1.0, label="EMA50")

    # VWAP (varsa)
    if "VWAP" in df.columns and df["VWAP"].notna().any():
        ax1.plot(x, df["VWAP"], linewidth=0.9, label="VWAP")

    ax1.set_title(f"{ticker} | prob_up={meta.get('prob_up','-'):.3f} | risk_flag={int(meta.get('risk_flag',0))} | ret_10={meta.get('ret_10','nan'):.3f} | ATR%={meta.get('atr_pct','nan'):.3f}")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Volume
    if "TOTAL TRADED VOLUME" in df.columns:
        ax2.bar(x, df["TOTAL TRADED VOLUME"].fillna(0).to_numpy())
        ax2.set_title("Volume", fontsize=9)
        ax2.grid(True, alpha=0.2)

    # RSI14
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))
    ax3.plot(x, rsi14, linewidth=1.0)
    ax3.axhline(70, linewidth=0.8)
    ax3.axhline(30, linewidth=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI(14)", fontsize=9)
    ax3.grid(True, alpha=0.2)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    ax4.plot(x, macd, linewidth=1.0, label="MACD")
    ax4.plot(x, sig, linewidth=1.0, label="Signal")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("MACD", fontsize=9)
    ax4.grid(True, alpha=0.2)

    # x-axis ticks
    tick_idx = np.linspace(0, len(df)-1, 8).astype(int)
    ax4.set_xticks(tick_idx)
    ax4.set_xticklabels([d.iloc[i].strftime("%Y-%m-%d") for i in tick_idx], rotation=30, ha="right", fontsize=8)

    fig.set_constrained_layout(True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _plot_market_overview(xu100_df: pd.DataFrame | None, out_png: Path):
    fig = plt.figure(figsize=(12, 5), dpi=150)
    ax = fig.add_subplot(1,1,1)
    if xu100_df is None or xu100_df.empty:
        ax.text(0.5, 0.5, "XU100 data not available", ha="center", va="center")
        ax.set_axis_off()
    else:
        xu100_df = xu100_df.sort_values("date").tail(260).copy()
        x = np.arange(len(xu100_df))
        ax.plot(x, xu100_df["close"], linewidth=1.0, label="XU100 Close")
        ax.plot(x, xu100_df["close"].ewm(span=20, adjust=False).mean(), linewidth=1.0, label="EMA20")
        ax.plot(x, xu100_df["close"].ewm(span=50, adjust=False).mean(), linewidth=1.0, label="EMA50")
        ax.set_title("Market Overview: XU100")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

        tick_idx = np.linspace(0, len(xu100_df)-1, 8).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([xu100_df["date"].iloc[i].strftime("%Y-%m-%d") for i in tick_idx],
                           rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_weekly_pdf(
    out_pdf: Path,
    stamp: str,
    all_equities_df: pd.DataFrame,
    picks_A: pd.DataFrame,
    picks_B: pd.DataFrame,
    overlap: pd.DataFrame,
    xu100_df: pd.DataFrame | None = None,
):
    out_pdf = Path(out_pdf)
    _ensure_dir(out_pdf.parent)
    img_dir = out_pdf.parent / f"_tmp_imgs_{stamp}"
    _ensure_dir(img_dir)

    # market overview image
    market_png = img_dir / "market.png"
    _plot_market_overview(xu100_df, market_png)

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    W, H = A4

    # --- Page 1: Cover/Market ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H-2.2*cm, f"Weekly BIST Report - {stamp}")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-3.0*cm, "Sections: Market Overview, Policy A Picks, Policy B Picks, Overlap")

    c.drawImage(ImageReader(str(market_png)), 2*cm, 3.2*cm, width=W-4*cm, height=H-7*cm, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # helper: draw one stock page
    def stock_page(row: pd.Series, section_title: str):
        ticker = row["ticker"]
        # hisse df
        g = all_equities_df[all_equities_df["ticker"] == ticker].copy()
        png = img_dir / f"{section_title}_{ticker}.png"

        # ATR% hesapla (panel meta için)
        close = g.sort_values("TRADE DATE")["CLOSING PRICE"]
        prev = close.shift(1)
        tr = np.maximum.reduce([
            (g["HIGHEST PRICE"] - g["LOWEST PRICE"]).abs().to_numpy(),
            (g["HIGHEST PRICE"] - prev).abs().to_numpy(),
            (g["LOWEST PRICE"] - prev).abs().to_numpy(),
        ])
        tr = pd.Series(tr, index=g.index)
        atr = tr.rolling(20).mean()
        atr_pct = float((atr / (g["CLOSING PRICE"] + 1e-9)).iloc[-1]) if len(g) else float("nan")

        meta = {
            "prob_up": float(row.get("prob_up", float("nan"))),
            "risk_flag": int(row.get("risk_flag", 0)),
            "ret_10": float(row.get("ret_10", float("nan"))),
            "atr_pct": atr_pct,
        }
        _plot_stock_panel(g, ticker, png, meta)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, H-2.2*cm, f"{section_title}: {ticker}")
        c.setFont("Helvetica", 10)

        # mini metrics box
        metrics = [
            ("prob_up", meta["prob_up"]),
            ("risk_flag", meta["risk_flag"]),
            ("ret_5", row.get("ret_5", np.nan)),
            ("ret_10", row.get("ret_10", np.nan)),
            ("ret_20", row.get("ret_20", np.nan)),
            ("ATR%", meta["atr_pct"]),
        ]
        y = H-3.0*cm
        for k, v in metrics:
            if isinstance(v, (int, np.integer)):
                c.drawString(2*cm, y, f"{k}: {int(v)}")
            else:
                try:
                    c.drawString(2*cm, y, f"{k}: {float(v):.4f}")
                except Exception:
                    c.drawString(2*cm, y, f"{k}: -")
            y -= 0.45*cm

        c.drawImage(ImageReader(str(png)), 2*cm, 2.3*cm, width=W-4*cm, height=H-7*cm, preserveAspectRatio=True, mask='auto')
        c.showPage()

    # --- Policy A pages ---
    if not picks_A.empty:
        for _, row in picks_A.iterrows():
            stock_page(row, "Policy A")

    # --- Policy B pages ---
    if not picks_B.empty:
        for _, row in picks_B.iterrows():
            stock_page(row, "Policy B")

    # --- Overlap page ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, H-2.2*cm, "Overlap (A ∩ B)")
    c.setFont("Helvetica", 10)

    if overlap.empty:
        c.drawString(2*cm, H-3.0*cm, "No overlap this week.")
    else:
        show_cols = ["ticker", "prob_up", "risk_flag", "ret_5", "ret_10", "ret_20"]
        show_cols = [x for x in show_cols if x in overlap.columns]
        t = overlap[show_cols].copy().sort_values("prob_up", ascending=False)

        y = H-3.2*cm
        c.drawString(2*cm, y, " | ".join(show_cols))
        y -= 0.6*cm
        for _, r in t.iterrows():
            vals = []
            for col in show_cols:
                v = r[col]
                if col == "ticker":
                    vals.append(str(v))
                elif col == "risk_flag":
                    vals.append(str(int(v)))
                else:
                    try:
                        vals.append(f"{float(v):.4f}")
                    except Exception:
                        vals.append("-")
            c.drawString(2*cm, y, " | ".join(vals))
            y -= 0.5*cm
            if y < 2.5*cm:
                c.showPage()
                y = H-2.5*cm

    c.showPage()
    c.save()

    # tmp png'leri bırakabilirsin; istersen sileriz.
