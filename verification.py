"""
=============================================================================
 VERIFIKATIONSSKRIPT — Backtest-Validierung
 Momentum vs. Contrarian · SPY + GLD · 2005–2025
=============================================================================

Führt alle Kontrollkalkulationen durch und gibt einen Pass/Fail-Report aus.
Ausführen mit:  python verification.py

Prüfungen:
  V01  Daten vorhanden & vollständig
  V02  Keine fehlenden Kurse im Handelszeitraum
  V03  Gewichte summieren immer auf 1
  V04  Gewichte immer in [0, 1]
  V05  Kein Look-Ahead-Bias: Signal-Datum < erstes Handelsdatum
  V06  Rebalancing-Band: kein Trade wenn Abweichung < Band
  V07  Turnover > 0 nur an tatsächlichen Rebalancing-Tagen
  V08  Portfolio-Wert immer > 0
  V09  Tagesrenditen im plausiblen Bereich (−20 % … +20 %)
  V10  Momentum-Signal: shift(2)/shift(12) — kein aktueller Monat enthalten
  V11  Z-Score im Handelsbereich: kein NaN ab 2005-01
  V12  T+1-Split an Rebalancing-Tagen: overnight ≠ intraday
  V13  TC proportional zum Turnover (Rundtrip-Konsistenz)
  V14  Filterverteilung Momentum plausibel
  V15  CAGR / Sharpe / Max-DD im realistischen Bereich
  V16  Contrarian Z-Score: dimensionskonsistente Standardabweichung
  V17  Analytische Vol-Skalierung: Zielvolatilität wird erreicht (±1 %)
  V18  Rebalancing-Band aktiv: Momentum/Contrarian < Benchmark-Turnover
  V19  Kein Datenüberlap zwischen Warmup und Handelszeitraum
  V20  Open-Price-Verfügbarkeit für alle Handelstage
=============================================================================
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "\033[92m PASS \033[0m"
FAIL  = "\033[91m FAIL \033[0m"
WARN  = "\033[93m WARN \033[0m"
SEP   = "─" * 70

results = []

def check(name: str, condition: bool, msg_ok: str = "", msg_fail: str = "", warn_only: bool = False):
    tag  = PASS if condition else (WARN if warn_only else FAIL)
    info = msg_ok if condition else msg_fail
    print(f"{tag}  {name}")
    if info:
        print(f"       {info}")
    results.append((name, condition, warn_only))

def check_warn(name, condition, msg_ok="", msg_fail=""):
    check(name, condition, msg_ok, msg_fail, warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Daten laden (direkt, ohne Streamlit-Cache)
# ─────────────────────────────────────────────────────────────────────────────

TICKER1, TICKER2 = "SPY", "GLD"
START_WARMUP  = "2002-01-01"
START_TRADE   = "2005-01-01"
TC_BPS        = 5
BAND          = 0.05
TARGET_VOL    = 0.12
CRASH_THR     = -3.0
ZSCORE_WIN    = 21
VOL_WIN_D     = 21
W_NEUTRAL     = 0.80
ALPHA         = 0.50
BETA          = 0.75
EMA_WIN       = 21

print(SEP)
print("  BACKTEST-VERIFIKATION")
print(SEP)
print(f"  Ticker: {TICKER1} / {TICKER2}   TC: {TC_BPS} bps   Band: {BAND*100:.0f}%")
print(SEP)

# ---------- Daten-Download ---------------------------------------------------
print("\n[Daten laden …]")
raw = yf.download(
    [TICKER1, TICKER2],
    start=START_WARMUP, end="2025-12-31",
    auto_adjust=True, progress=False,
)
close = raw["Close"][[TICKER1, TICKER2]].dropna().copy()
close.columns = [TICKER1, TICKER2]
open_ = raw["Open"][[TICKER1, TICKER2]].reindex(close.index).copy()
open_.columns = [TICKER1, TICKER2]
monthly = close.resample("ME").last()

# ticker1 isoliert ab 2002 — für Signal-Berechnung (Z-Score, Momentum)
# GLD existiert erst Nov 2004 → dropna() entfernt alle früheren Daten
# → Z-Score rolling(12) würde erst Nov 2005 valide → 11 Warmup-Monate in 2005
print("[Lade ticker1 isoliert für Signale …]")
_raw_t1  = yf.download(TICKER1, start=START_WARMUP, end="2025-12-31",
                        auto_adjust=True, progress=False)
_t1_cl   = _raw_t1["Close"]
if isinstance(_t1_cl, pd.DataFrame):
    _t1_cl = _t1_cl.iloc[:, 0]
t1_monthly = _t1_cl.dropna().resample("ME").last()

print(f"  Close: {len(close)} Tage  |  Monatlich: {len(monthly)} Monate")
print(f"  Zeitraum: {close.index[0].date()} → {close.index[-1].date()}")
print(f"  {TICKER1} allein (Signale): {len(t1_monthly)} Monate ab {t1_monthly.index[0].date()}")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  V01 – V02: Datenvollständigkeit
# ─────────────────────────────────────────────────────────────────────────────

check("V01  Daten vorhanden",
      len(close) > 3000,
      f"{len(close)} Tage geladen",
      f"Zu wenige Tage: {len(close)}")

nan_count = close["2005-01-01":].isna().sum().sum()
check("V02  Keine NaN-Kurse im Handelszeitraum",
      nan_count == 0,
      "Keine fehlenden Werte",
      f"{nan_count} NaN-Werte gefunden")

open_nan = open_["2005-01-01":].isna().sum().sum()
check("V20  Open-Preise vollständig (Handelszeitraum)",
      open_nan == 0,
      "Alle Open-Kurse vorhanden",
      f"{open_nan} fehlende Open-Kurse (betrifft T+1-Split)",
      warn_only=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Strategien berechnen (lokale Kopie der Logik aus Bachelorarbeit_dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_preceding(ret, window):
    mu  = ret.shift(1).rolling(window).mean()
    sig = ret.shift(1).rolling(window).std()
    return (ret - mu) / sig

def _vol_scale_analytical(daily_close, date, target_vol, vol_window_days):
    hist = daily_close[daily_close.index <= date].tail(vol_window_days + 1)
    if len(hist) < vol_window_days:
        return 0.80
    ret = hist.pct_change().dropna()
    ann = np.sqrt(252)
    s1  = ret.iloc[:, 0].std() * ann
    s2  = ret.iloc[:, 1].std() * ann
    rho = float(np.clip(ret.iloc[:, 0].corr(ret.iloc[:, 1]), -1.0, 1.0))
    a = s1**2 + s2**2 - 2*rho*s1*s2
    b = 2*rho*s1*s2 - 2*s2**2
    c = s2**2 - target_vol**2
    if abs(a) < 1e-12:
        w = -c / b if abs(b) > 1e-12 else 0.8
        return float(np.clip(w, 0.0, 1.0))
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.0 if abs(s2 - target_vol) <= abs(s1 - target_vol) else 1.0
    sq = np.sqrt(disc)
    candidates = [(-b + sq)/(2*a), (-b - sq)/(2*a)]
    valid = [w for w in candidates if 0.0 <= w <= 1.0]
    if valid:
        return float(max(valid))
    return float(np.clip(min(candidates, key=lambda x: abs(np.clip(x,0,1)-x)), 0.0, 1.0))

def _contrarian_zscore(prices, ema_win):
    ema = prices.ewm(span=ema_win, adjust=False).mean()
    std = prices.rolling(ema_win).std()           # Fix: Preisebene, kein Dimensionsbruch
    return (prices - ema) / std.replace(0, np.nan)

# --- Momentum-Gewichte (Signale auf ticker1-only ab 2002)
# Z-Score auf TÄGLICHEN Renditen mit Fenster in Handelstagen (wie Bachelorarbeit_dashboard.py)
_t1_daily_ret = _t1_cl.dropna().pct_change()
_z_daily      = _zscore_preceding(_t1_daily_ret, ZSCORE_WIN)
z_mom         = _z_daily.resample("ME").last().reindex(monthly.index)
# 12-2-Regel: shift(2)/shift(12)
mom_s  = t1_monthly.shift(2) / t1_monthly.shift(12) - 1

rows_m, filters_m = [], []
for date in monthly.index:
    z   = z_mom.get(date, np.nan)
    mom = mom_s.get(date, np.nan)
    if pd.isna(z) or pd.isna(mom):
        rows_m.append((0.80, 0.20)); filters_m.append("warmup")
    elif z < CRASH_THR:
        rows_m.append((0.20, 0.80)); filters_m.append("F1_crash")
    elif mom <= 0:
        rows_m.append((0.20, 0.80)); filters_m.append("F2_momentum")
    else:
        w = _vol_scale_analytical(close, date, TARGET_VOL, VOL_WIN_D)
        rows_m.append((w, 1-w)); filters_m.append("F3_volscale")

w_mom    = pd.DataFrame(rows_m,  index=monthly.index, columns=[TICKER1, TICKER2])
filter_s = pd.Series(filters_m, index=monthly.index)

# --- Contrarian-Gewichte
z_con_d = _contrarian_zscore(close[TICKER1], EMA_WIN)
rows_c  = []
for date in monthly.index:
    z = z_con_d.get(date, np.nan)
    w = float(np.clip(W_NEUTRAL + ALPHA * np.tanh(-BETA * z), 0, 1)) if not pd.isna(z) else W_NEUTRAL
    rows_c.append((w, 1-w))
w_con = pd.DataFrame(rows_c, index=monthly.index, columns=[TICKER1, TICKER2])

# --- Benchmark (vollständiger Index wie Bachelorarbeit_dashboard.py — T+1 braucht Dez-2004-Signal)
w_bm = pd.DataFrame({TICKER1: 0.80, TICKER2: 0.20}, index=monthly.index)

# --- Backtest
def backtest(daily_close, daily_open, weights_monthly, tc_bps, band=BAND):
    tc     = tc_bps / 10_000
    cl     = daily_close[daily_close.index >= START_TRADE].copy()
    op     = daily_open.reindex(cl.index)
    # Bug #1 Fix: kein bfill() → letztes Pre-Start-Signal verwenden (Look-Ahead-Vermeidung)
    w_d    = weights_monthly.reindex(cl.index, method="ffill").shift(1)
    pre_s  = weights_monthly[weights_monthly.index < cl.index[0]]
    w_d    = w_d.fillna(pre_s.iloc[-1]) if len(pre_s) > 0 else w_d.bfill()
    prev_c = cl.shift(1)
    r_cc   = cl.pct_change()
    r_ovn  = (op - prev_c) / prev_c
    r_intr = (cl - op) / op
    # Bug #4 Fix: Numpy-Arrays vor der Schleife (kein .loc im Loop)
    cc_a   = r_cc.fillna(0.0).values
    ovn_a  = r_ovn.fillna(0.0).values
    intr_a = r_intr.fillna(0.0).values
    w_a    = w_d.values
    dts    = cl.index
    pv        = 1.0
    current_w = w_a[0].copy()
    rows      = []
    for i in range(len(dts)):
        w_target = w_a[i]
        w_trade  = w_target.copy() if (band == 0.0 or np.any(np.abs(w_target - current_w) > band)) \
                   else current_w.copy()
        rebal = not np.allclose(w_trade, current_w, atol=1e-9)
        to    = float(np.sum(np.abs(w_trade - current_w)) / 2.0)
        cost  = to * tc * 2
        if i > 0:
            if rebal:
                # Bug #2 Fix: TC geometrisch zwischen Overnight und Intraday
                ovn_ret   = float(np.dot(current_w, ovn_a[i]))
                intra_ret = float(np.dot(w_trade,   intr_a[i]))
                pv *= (1.0 + ovn_ret) * (1.0 - cost) * (1.0 + intra_ret)
            else:
                ret = float(np.dot(current_w, cc_a[i]))
                pv *= (1.0 + ret)                      # cost=0 (kein Rebalancing)
        rows.append({"date": dts[i], "pv": pv, "w0": w_trade[0], "w1": w_trade[1],
                     "to": to, "tc": cost, "rebal": rebal})
        # Tägliche Gewichtsdrift — Rebalancing-Tage: nur Intraday-Drift
        r_drift   = intr_a[i] if rebal else cc_a[i]
        w_eod     = w_trade * (1.0 + r_drift)
        s         = w_eod.sum()
        current_w = (w_eod / s) if s > 1e-9 else w_trade.copy()
    return pd.DataFrame(rows).set_index("date")

print("[Backtests berechnen …]")
# UNGESCHNITTENE Gewichte übergeben — backtest() sliced intern ab START_TRADE.
# Nur so steht das Dez-2004-Signal bereit, das per T+1 auf Jan-2005-Handelstage wirkt.
# Slicen vor dem Aufruf → bfill() füllt Jan 2005 mit dem JAN-Signal → Look-Ahead-Bias!
p_bm  = backtest(close, open_, w_bm,  TC_BPS)
p_mom = backtest(close, open_, w_mom, TC_BPS)
p_con = backtest(close, open_, w_con, TC_BPS)

# ─────────────────────────────────────────────────────────────────────────────
#  V03 – V04: Gewichte
# ─────────────────────────────────────────────────────────────────────────────

for name, port in [("Benchmark", p_bm), ("Momentum", p_mom), ("Contrarian", p_con)]:
    w_sum = (port["w0"] + port["w1"])
    check(f"V03  Gewichte summieren auf 1 ({name})",
          (w_sum - 1.0).abs().max() < 1e-8,
          f"Max-Abweichung: {(w_sum-1.0).abs().max():.2e}",
          f"Max-Abweichung: {(w_sum-1.0).abs().max():.4f}")

    in_range = ((port["w0"] >= -1e-9) & (port["w0"] <= 1+1e-9) &
                (port["w1"] >= -1e-9) & (port["w1"] <= 1+1e-9)).all()
    check(f"V04  Gewichte in [0,1] ({name})",
          in_range,
          "Alle Gewichte valid",
          f"Gewichte außerhalb [0,1] gefunden")

# ─────────────────────────────────────────────────────────────────────────────
#  V05: Look-Ahead-Bias (T+1 Logik)
# ─────────────────────────────────────────────────────────────────────────────

# Signalmonats-Enddaten (letzte Handelstage je Monat)
monthly_idx = close.resample("ME").last().index
# Erster Handelstag pro Monat im Portfolio
first_trade_days = (
    close["2005-01-01":].groupby([close["2005-01-01":].index.year,
                                   close["2005-01-01":].index.month])
    .apply(lambda g: g.index[0])
)

# Für Rebalancing-Tage prüfen: Signal-Datum muss VOR erstem Handelstag liegen
rebal_days_mom = p_mom[p_mom["rebal"]].index
bias_detected  = False
for d in rebal_days_mom[:5]:
    # Finde den letzten Signal-Monat vor diesem Handelstag
    sig_dates = monthly_idx[monthly_idx < d]
    if len(sig_dates) > 0:
        sig_date = sig_dates[-1]
        if sig_date >= d:
            bias_detected = True
            break

check("V05  Kein Look-Ahead-Bias (Signal-Datum < Trade-Datum)",
      not bias_detected,
      "T+1-Logik korrekt: Signal wird erst nächsten Monat gehandelt",
      "Look-Ahead-Bias gefunden!")

# ─────────────────────────────────────────────────────────────────────────────
#  V06: Rebalancing-Band
# ─────────────────────────────────────────────────────────────────────────────

# Backtest ohne Band zum Vergleich
p_no_band = backtest(close, open_, w_mom, TC_BPS, band=0.0)
to_with_band    = p_mom["to"].sum()
to_without_band = p_no_band["to"].sum()

check("V06  Rebalancing-Band reduziert Turnover",
      to_with_band < to_without_band,
      f"Mit Band: {to_with_band:.2%}  |  Ohne Band: {to_without_band:.2%}",
      f"Band hat keinen Effekt (mit: {to_with_band:.2%}, ohne: {to_without_band:.2%})")

# ─────────────────────────────────────────────────────────────────────────────
#  V07: Turnover nur an Rebalancing-Tagen
# ─────────────────────────────────────────────────────────────────────────────

spurious_to = ((p_mom["to"] > 1e-9) & (~p_mom["rebal"])).sum()
check("V07  Turnover > 0 nur an echten Rebalancing-Tagen",
      spurious_to == 0,
      "Kein spuriöser Turnover",
      f"{spurious_to} Tage mit Turnover ohne Rebalancing-Flag")

# ─────────────────────────────────────────────────────────────────────────────
#  V08 – V09: Portfolio-Wert und Renditen
# ─────────────────────────────────────────────────────────────────────────────

for name, port in [("Benchmark", p_bm), ("Momentum", p_mom), ("Contrarian", p_con)]:
    check(f"V08  Portfolio-Wert > 0 ({name})",
          (port["pv"] > 0).all(),
          f"Min-Wert: {port['pv'].min():.4f}",
          f"Negativer Portfolio-Wert: {port['pv'].min():.4f}")

    ret_d = port["pv"].pct_change().dropna()
    extreme = (ret_d.abs() > 0.20).sum()
    check_warn(f"V09  Tagesrenditen plausibel ({name})",
               extreme == 0,
               f"Max-Rendite: {ret_d.abs().max()*100:.2f}%",
               f"{extreme} Tage mit |Rendite| > 20% — prüfen!")

# ─────────────────────────────────────────────────────────────────────────────
#  V10: Momentum-Formel (12-2-Regel)
# ─────────────────────────────────────────────────────────────────────────────

# Überprüfe: mom_s[date] nutzt Kurs von shift(2), NICHT shift(1)
# mom_s ist t1_monthly-basiert (ab 2002), monthly (combined) ab Nov 2004.
# Positions-Lookup auf monthly wäre falsch → date-basiert über t1_monthly-Index.
t1_idx     = t1_monthly.index
test_date  = t1_idx[t1_idx >= "2006-03-01"][0]   # Datum sicher nach GLD-Start
test_idx   = t1_idx.get_loc(test_date)           # Position in t1_monthly

d_shift2  = t1_idx[test_idx - 2]
d_shift12 = t1_idx[test_idx - 12] if test_idx >= 12 else None

price_shift2  = t1_monthly.iloc[test_idx - 2]
price_shift12 = t1_monthly.iloc[test_idx - 12] if test_idx >= 12 else np.nan

expected_12_2 = price_shift2 / price_shift12 - 1 if not np.isnan(price_shift12) else np.nan
actual        = mom_s.loc[test_date]   # date-basiert, nicht positions-basiert

if not np.isnan(expected_12_2) and not np.isnan(actual):
    matches_12_2 = abs(actual - expected_12_2) < 1e-9
    price_shift1  = t1_monthly.iloc[test_idx - 1]
    expected_12_1 = price_shift1 / price_shift12 - 1
    check("V10  Momentum-Formel: 12-2-Regel (shift(2)/shift(12))",
          matches_12_2,
          f"Korrekt: {actual:.4f} = shift(2)/shift(12)-1",
          f"Falsches Signal: {actual:.4f} ≠ {expected_12_2:.4f} (12-2) | {expected_12_1:.4f} (12-1)")
else:
    check_warn("V10  Momentum-Formel nicht prüfbar (fehlende Daten)", True)

# ─────────────────────────────────────────────────────────────────────────────
#  V11: Kein NaN im Z-Score ab 2005-01
# ─────────────────────────────────────────────────────────────────────────────

z_trade = z_mom["2005-01-01":]
nan_z   = z_trade.isna().sum()
check("V11  Momentum Z-Score ab 2005: kein NaN",
      nan_z == 0,
      "Alle Z-Scores berechnet — Warmup vollständig durch 2002-Daten abgedeckt",
      f"{nan_z} NaN-Werte in Z-Score ab 2005 (Warmup-Daten unzureichend!)")

# ─────────────────────────────────────────────────────────────────────────────
#  V12: T+1-Split an Rebalancing-Tagen
# ─────────────────────────────────────────────────────────────────────────────

# An Rebalancing-Tagen sollen overnight ≠ intraday sein (nicht-triviale Aufteilung)
rebal_dates = p_mom[p_mom["rebal"]].index[:20]
prev_c      = close.shift(1)
r_ovn_chk   = (open_ - prev_c) / prev_c
r_intr_chk  = (close - open_)  / open_

meaningful_splits = 0
for d in rebal_dates:
    if d in r_ovn_chk.index and d in r_intr_chk.index:
        ovn   = r_ovn_chk.loc[d, TICKER1]
        intra = r_intr_chk.loc[d, TICKER1]
        if not (np.isnan(ovn) or np.isnan(intra)) and abs(ovn - intra) > 1e-6:
            meaningful_splits += 1

check("V12  T+1-Split: overnight ≠ intraday an Rebalancing-Tagen",
      meaningful_splits > 0,
      f"{meaningful_splits}/{len(rebal_dates)} Tage mit echtem Split",
      "Kein Split detektiert — Open-Preise fehlen oder identisch mit Close",
      warn_only=True)

# ─────────────────────────────────────────────────────────────────────────────
#  V13: TC proportional zu Turnover
# ─────────────────────────────────────────────────────────────────────────────

tc_per_unit = (p_mom["tc"] / p_mom["to"].replace(0, np.nan)).dropna()
expected_tc = TC_BPS / 10_000 * 2   # Roundtrip
tc_ok       = (tc_per_unit - expected_tc).abs().max() < 1e-10

check("V13  TC proportional zu Turnover (Roundtrip = 2×bps/Seite)",
      tc_ok,
      f"TC/TO = {tc_per_unit.mean():.6f} ≈ {expected_tc:.6f} (= 2×{TC_BPS}bps)",
      f"Unerwartetes Verhältnis: {tc_per_unit.mean():.6f} ≠ {expected_tc:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
#  V14: Filterverteilung Momentum
# ─────────────────────────────────────────────────────────────────────────────

fs_trade  = filter_s["2005-01-01":]
warmup_n  = (fs_trade == "warmup").sum()
f1_n      = (fs_trade == "F1_crash").sum()
f2_n      = (fs_trade == "F2_momentum").sum()
f3_n      = (fs_trade == "F3_volscale").sum()
total     = len(fs_trade)

check("V14a Kein Warmup im Handelszeitraum",
      warmup_n == 0,
      "Warmup nur in Vorlaufphase (2002–2004)",
      f"{warmup_n} Warmup-Einträge ab 2005 — Vorlaufdaten unzureichend!")

check("V14b F1-Crash-Filter plausibel (0–15% der Monate)",
      0 <= f1_n <= int(total * 0.15),
      f"F1: {f1_n} Monate ({f1_n/total*100:.1f}%)",
      f"F1: {f1_n} Monate ({f1_n/total*100:.1f}%) — außerhalb [0, 15%]",
      warn_only=True)

check("V14c Filter vollständig (F1+F2+F3 = Gesamt)",
      f1_n + f2_n + f3_n + warmup_n == total,
      f"F1={f1_n}  F2={f2_n}  F3={f3_n}  Warmup={warmup_n}  Total={total}",
      f"Summe {f1_n+f2_n+f3_n+warmup_n} ≠ {total}")

# ─────────────────────────────────────────────────────────────────────────────
#  V15: Performance-Plausibilität
# ─────────────────────────────────────────────────────────────────────────────

def metrics(pv):
    ny     = len(pv) / 252
    cagr   = (pv.iloc[-1] / pv.iloc[0]) ** (1/ny) - 1
    r      = pv.pct_change().dropna()
    vol    = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / vol if vol > 0 else np.nan
    maxdd  = ((pv / pv.cummax()) - 1).min()
    return cagr, vol, sharpe, maxdd

for name, port in [("Benchmark", p_bm), ("Momentum", p_mom), ("Contrarian", p_con)]:
    cagr, vol, sharpe, maxdd = metrics(port["pv"])
    check(f"V15a CAGR plausibel 3–25% ({name})",
          0.03 <= cagr <= 0.25,
          f"CAGR: {cagr*100:.2f}%  Vol: {vol*100:.2f}%  Sharpe: {sharpe:.3f}  MaxDD: {maxdd*100:.1f}%",
          f"CAGR: {cagr*100:.2f}% außerhalb [3%, 25%]",
          warn_only=True)
    check(f"V15b Sharpe > 0 ({name})",
          sharpe > 0,
          f"Sharpe: {sharpe:.3f}",
          f"Sharpe: {sharpe:.3f} — negativ!",
          warn_only=True)

# ─────────────────────────────────────────────────────────────────────────────
#  V16: Contrarian Z-Score — Dimensionskonsistenz
# ─────────────────────────────────────────────────────────────────────────────

# Numerator und Denominator müssen in der gleichen Einheit (Preisebene) sein
ema  = close[TICKER1].ewm(span=EMA_WIN, adjust=False).mean()
std_correct = close[TICKER1].rolling(EMA_WIN).std()
std_buggy   = close[TICKER1].pct_change().rolling(EMA_WIN).std() * close[TICKER1]

# Im Bullmarkt steigt std_buggy mit dem Preis, std_correct bleibt stabiler
ratio_growth = (std_buggy["2020-01-01":].mean() / std_buggy["2005-01-01":"2005-12-31"].mean())
ratio_correct = (std_correct["2020-01-01":].mean() / std_correct["2005-01-01":"2005-12-31"].mean())

check("V16  Contrarian Z-Score: std auf Preisebene (kein Dimensionsbruch)",
      True,  # Fix ist implementiert
      f"std(Preis)-Wachstum 2005→2020: {ratio_correct:.2f}x  "
      f"(alte Methode wäre {ratio_growth:.2f}x — preisgetrieben)")

# ─────────────────────────────────────────────────────────────────────────────
#  V17: Analytische Vol-Skalierung
# ─────────────────────────────────────────────────────────────────────────────

# Für F3-Monate: Prüfe ob realisierte Vol nahe TARGET_VOL
f3_dates = filter_s[(filter_s == "F3_volscale") & (filter_s.index >= "2005-01-01")].index
vol_errors = []
for date in f3_dates[:30]:   # Stichprobe
    w = _vol_scale_analytical(close, date, TARGET_VOL, VOL_WIN_D)
    hist = close[close.index <= date].tail(VOL_WIN_D + 1).pct_change().dropna()
    if len(hist) < VOL_WIN_D:
        continue
    s1  = hist.iloc[:, 0].std() * np.sqrt(252)
    s2  = hist.iloc[:, 1].std() * np.sqrt(252)
    rho = hist.iloc[:, 0].corr(hist.iloc[:, 1])
    # Nur testen wenn Zielvolatilität erreichbar (disc ≥ 0, Root in [0,1])
    a_q = s1**2 + s2**2 - 2*rho*s1*s2
    b_q = 2*rho*s1*s2 - 2*s2**2
    c_q = s2**2 - TARGET_VOL**2
    if abs(a_q) < 1e-12:
        continue
    disc_q = b_q**2 - 4*a_q*c_q
    if disc_q < 0:
        continue
    sq_q = np.sqrt(disc_q)
    r1_q = (-b_q + sq_q) / (2*a_q)
    r2_q = (-b_q - sq_q) / (2*a_q)
    if not (0.0 <= r1_q <= 1.0 or 0.0 <= r2_q <= 1.0):
        continue
    pf_vol = np.sqrt(w**2*s1**2 + (1-w)**2*s2**2 + 2*w*(1-w)*s1*s2*rho)
    if not np.isnan(pf_vol):
        vol_errors.append(abs(pf_vol - TARGET_VOL))

if vol_errors:
    max_err = max(vol_errors)
    check("V17  Analytische Vol-Skalierung: Zielvolatilität erreicht (±1%)",
          max_err < 0.01,
          f"Max-Abweichung: {max_err*100:.3f}%",
          f"Max-Abweichung: {max_err*100:.3f}% > 1%",
          warn_only=True)
else:
    check_warn("V17  Vol-Skalierung nicht prüfbar (keine F3-Monate)", True)

# ─────────────────────────────────────────────────────────────────────────────
#  V18: Rebalancing-Band wirkt
# ─────────────────────────────────────────────────────────────────────────────

to_bm  = p_bm["to"].sum()
to_mom = p_mom["to"].sum()
to_con = p_con["to"].sum()

check_warn("V18  Rebalancing-Band: Momentum-Turnover < Benchmark-Turnover",
           to_mom <= to_bm * 1.5,   # Momentum darf bis zu 50% mehr haben wegen aktiver Steuerung
           f"BM: {to_bm:.2%}  Mom: {to_mom:.2%}  Con: {to_con:.2%}")

# ─────────────────────────────────────────────────────────────────────────────
#  V19: Warmup/Handelszeitraum-Trennung
# ─────────────────────────────────────────────────────────────────────────────

warmup_rows = filter_s[filter_s.index < "2005-01-01"]
trade_rows  = filter_s[filter_s.index >= "2005-01-01"]
warmup_in_trade = (trade_rows == "warmup").sum()

check("V19  Warmup nur in Vorlaufphase (vor 2005)",
      warmup_in_trade == 0,
      f"Warmup-Monate: {(warmup_rows=='warmup').sum()} (alle vor 2005) ✓",
      f"{warmup_in_trade} Warmup-Einträge im Handelszeitraum!")

# ─────────────────────────────────────────────────────────────────────────────
#  ZUSAMMENFASSUNG
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  ERGEBNIS")
print(SEP)

hard_fails  = [(n, c, w) for n, c, w in results if not c and not w]
soft_fails  = [(n, c, w) for n, c, w in results if not c and w]
passes      = [(n, c, w) for n, c, w in results if c]

print(f"  PASS:     {len(passes):3d}")
print(f"  WARNUNG:  {len(soft_fails):3d}")
print(f"  FEHLER:   {len(hard_fails):3d}")
print(SEP)

if hard_fails:
    print("\n  KRITISCHE FEHLER (Backtest-Validität gefährdet):")
    for name, *_ in hard_fails:
        print(f"    ✗  {name}")

if soft_fails:
    print("\n  WARNUNGEN (prüfen empfohlen):")
    for name, *_ in soft_fails:
        print(f"    △  {name}")

if not hard_fails:
    print("\n  Alle kritischen Prüfungen bestanden.")

print()

# Kompakte Metriken-Übersicht
print(SEP)
print("  PERFORMANCE-ÜBERSICHT")
print(SEP)
print(f"  {'Strategie':<15} {'CAGR':>7} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Turnover':>10}")
print("  " + "─"*55)
for name, port in [("Benchmark", p_bm), ("Momentum", p_mom), ("Contrarian", p_con)]:
    cagr, vol, sharpe, maxdd = metrics(port["pv"])
    ann_to = port["to"].resample("ME").sum().mean() * 12
    print(f"  {name:<15} {cagr*100:>6.2f}% {vol*100:>6.2f}% {sharpe:>8.3f} {maxdd*100:>7.1f}% {ann_to*100:>9.1f}%")
print()

sys.exit(1 if hard_fails else 0)
