# =============================================================================
#  INTERAKTIVES BACKTESTING-DASHBOARD
#  Momentum vs. Contrarian · SPY + GLD · 2005–2025
#  Bachelorarbeit · Lukas Kaska
#
#  Starten:  streamlit run Bachelorarbeit_dashboard.py
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        INHALTSVERZEICHNIS                               │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  ★ FÜR DIE BACHELORARBEIT WICHTIG — Kernlogik                          │
# │  ─────────────────────────────────────────────────────────────────────  │
# │  §1  KONFIGURATION & FARBSCHEMA          ca. Zeile  45                  │
# │  §2  DATENBESCHAFFUNG                    ca. Zeile  91                  │
# │       load_prices()      Yahoo Finance, Close + Open ab 2002            │
# │       load_t1_prices()   Nur Asset 1 isoliert (für Signale ab 2002)     │
# │  §3  BACKTEST-ENGINE                     ca. Zeile 159                  │
# │       _backtest()        T+1-Split · Gewichtsdrift · TC · Band          │
# │  §4  HILFSFUNKTIONEN                     ca. Zeile 267                  │
# │       _zscore_preceding()    Rollierender Z-Score (vorausschauend)      │
# │       _vol_scale_weight()    Analytische Vol-Skalierung (Mitternacht)   │
# │  §5  STRATEGIE A: MOMENTUM               ca. Zeile 331                  │
# │       _momentum_weights_cached()  F1-Crash / F2-Momentum / F3-VolScale  │
# │  §6  STRATEGIE B: CONTRARIAN             ca. Zeile 404                  │
# │       _contrarian_zscore()        EMA-basierter Z-Score                 │
# │       _contrarian_weights_cached() tanh-Allokationsformel               │
# │  §7  PERFORMANCE-METRIKEN                ca. Zeile 458                  │
# │       calc_metrics()     CAGR · Sharpe · Sortino · MaxDD · DDDauer · TO │
# │                                                                         │
# │  ✦ VISUALISIERUNG & APP-INFRASTRUKTUR — weniger kritisch                │
# │  ─────────────────────────────────────────────────────────────────────  │
# │  §8  PLOTLY-CHARTS                       ca. Zeile 510                  │
# │       Equity · Drawdown · Allokation · Rolling Sharpe · Z-Score         │
# │  §9  MONATSTABELLE                       ca. Zeile 748                  │
# │  §10 LIVE-DASHBOARD  main()              ca. Zeile 813                  │
# │       §10a Sidebar-Parameter                                            │
# │       §10b Datenberechnung & Backtest-Aufruf                            │
# │       §10c KPI-Tabelle                                                  │
# │       §10d Charts & Tabellen anzeigen                                   │
# │       §10e Analyse-Export (Markdown-Bericht)                            │
# │       §10f Logik-Check W1–W22 (Aufruf)                                  │
# │  §11 LOGIK-CHECKS  _run_logic_checks()   ca. Zeile 798                  │
# │       Gruppen A–F · W1–W22 · automatisierte Selbstprüfung               │
# └─────────────────────────────────────────────────────────────────────────┘

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm

# ─────────────────────────────────────────────────────────────────────────────
#  SEITEN-KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Momentum vs. Contrarian",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-container { text-align: center; }
    div[data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 10px; }
    h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  REGIME-DEFINITIONEN (für Charts)
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = [
    ("2007-10-01", "2009-03-31", "GFC 2007–09",    "rgba(255,180,180,0.25)"),
    ("2020-02-01", "2020-04-30", "COVID 2020",      "rgba(255,220,160,0.25)"),
    ("2022-01-01", "2022-12-31", "Bärenmarkt 2022", "rgba(180,210,240,0.25)"),
]

COLORS = {
    "Benchmark":  "#555555",
    "Momentum":   "#1f77b4",
    "Contrarian": "#d62728",
}

# =============================================================================
# §2  DATENBESCHAFFUNG  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Wir laden adjustierte Tagespreise (Close + Open) von Yahoo Finance.
# "Adjustiert" = Dividenden und Splits sind bereits eingerechnet → korrekte
# Gesamtrendite (Total Return). Zeitraum: 2002–2025.
#
# WARUM OPEN-PREISE?
#   Der Backtest führt Umschichtungen am nächsten Handelstag aus (T+1).
#   Dabei wird der Return aufgeteilt:
#     Overnight (Close_T → Open_{T+1}): mit alten Gewichten
#     Intraday  (Open_{T+1} → Close_{T+1}): mit neuen Gewichten
#   Ohne Open-Preise wäre dieser Split nicht möglich → Look-Ahead-Bias.
#
# WARUM AB 2002?
#   GLD (Gold-ETF) existiert erst ab November 2004. Der Z-Score für Momentum
#   braucht 12 Monate Vorlauf. Ohne Daten ab 2002 wären die ersten 11 Monate
#   von 2005 "Warmup" (kein valides Signal) → Verzerrung der Backtests.
#   Lösung: Asset 1 (SPY) separat ab 2002 für die Signalberechnung laden.
# =============================================================================

@st.cache_data(show_spinner="Lade Marktdaten von Yahoo Finance …")
def load_prices(ticker1="SPY", ticker2="GLD"):
    """Lädt Tagespreise ab 2002.

    Gibt (close, open_) zurück:
      close  – adjustierte Schlusskurse (Total-Return-Basis, für Portfolio-Renditen)
      open_  – adjustierte Eröffnungskurse (für T+1-Ausführungssplit an Rebalancing-Tagen)

    Beide Serien nutzen auto_adjust=True, damit Close/Open konsistent skaliert sind
    und intraday/overnight Renditen korrekt addieren.
    """
    raw = yf.download(
        [ticker1, ticker2],
        start="2002-01-01", end="2025-12-31",
        auto_adjust=True, progress=False,
    )
    cols  = [ticker1, ticker2]
    close = raw["Close"][cols].dropna().copy()
    close.columns = cols
    open_ = raw["Open"][cols].reindex(close.index).copy()
    open_.columns = cols
    return close, open_


@st.cache_data(show_spinner=False)
def get_monthly(daily):
    return daily.resample("ME").last()


@st.cache_data(show_spinner=False)
def load_t1_prices(ticker1="SPY"):
    """Lädt nur ticker1 ab 2002 – isoliert für Signal-Berechnung.

    Problem: GLD (ticker2) existiert erst ab Nov 2004. load_prices() kombiniert
    beide Ticker mit dropna() → alle Daten vor Nov 2004 fallen weg → Z-Score mit
    rolling(12) wird erst Nov 2005 valide → 11 Warmup-Monate in 2005.

    Lösung: ticker1 separat laden → volle Warmup-Periode ab 2002 verfügbar.
    """
    raw = yf.download(ticker1, start="2002-01-01", end="2025-12-31",
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker1)


# =============================================================================
# §3  BACKTEST-ENGINE  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Die Backtest-Engine simuliert täglich, wie sich das Portfolio entwickelt.
# Sie berücksichtigt drei realitätsnahe Mechanismen:
#
# 1. T+1-AUSFÜHRUNGSSPLIT (kein Look-Ahead-Bias)
#    Signale vom Monatsende M werden erst am ersten Tag von M+1 gehandelt.
#    An Rebalancing-Tagen wird die Rendite aufgeteilt:
#      R_tag = (1 + dot(w_alt, r_overnight)) × (1 + dot(w_neu, r_intraday)) − 1
#
# 2. GEWICHTSDRIFT
#    Nach jedem Tag driften die Gewichte mit den Marktrenditen:
#      w_eod = w_trade × (1 + r)   dann normiert auf Summe=1
#    Der Rebalancing-Trigger vergleicht ZIEL vs. AKTUELL GEDRIFTETE Gewichte.
#    → Auch der "statische" Benchmark zahlt TC, wenn er zu weit drift.
#
# 3. REBALANCING-BAND
#    Nur rebalancieren wenn |w_ziel − w_aktuell| > Band für mind. ein Asset.
#    → Reduziert unnötige Transaktionskosten ("Whipsaw-TC").
#
# TRANSAKTIONSKOSTEN-FORMEL:
#    to   = Σ|Δw| / 2          (einseitiger Umsatz)
#    cost = to × tc × 2        (Hin- und Rückseite = Roundtrip)
#    → entspricht:  Σ|Δw| × tc  pro Rebalancing
# =============================================================================

def _backtest(daily_close, daily_open, weights_monthly, tc_bps, band=0.05, start="2005-01-01"):
    """Backtest-Engine mit Gewichtsdrift, Rebalancing-Band und T+1-Ausführungssplit.

    Gewichtsdrift:
      Die tatsächlichen Portfolio-Gewichte driften täglich mit den Marktrenditen.
      Der Rebalancing-Trigger vergleicht den ZIELWERT mit den AKTUELL GEDRIFTETEN
      Gewichten — nicht mehr mit dem letzten Zielwert. Damit entstehen auch beim
      Benchmark (80/20) realistische Transaktionskosten, wenn SPY/GLD aus dem
      Zielband driftet.

    Rebalancing-Band (band > 0):
      Kein Trade solange |w_ziel - w_aktuell| ≤ band für alle Assets.

    T+1-Ausführungssplit:
      An Rebalancing-Tagen:
        • Overnight (Close_{T-1} → Open_T): gedriftete Gewichte
        • Intraday  (Open_T     → Close_T): neue Zielgewichte
        • TC wird geometrisch zwischen Overnight und Intraday abgezogen:
          PV *= (1 + r_ovn) × (1 − cost) × (1 + r_intra)
      Normale Tage: Close-to-Close mit gedrifteten Gewichten.

    TC-Formel: cost = to * tc * 2
      to = |Δw| / 2 (einseitiger Umschlag) · tc = bps/Seite · ×2 = Roundtrip.
    """
    tc     = tc_bps / 10_000
    close  = daily_close[daily_close.index >= start].copy()
    open_  = daily_open.reindex(close.index)

    # Monatliche Zielgewichte auf jeden Handelstag ausweiten und um 1 Tag nach
    # vorne schieben (T+1): Signal von Monatsende M wird erst ab M+1 gehandelt.
    # Bug #1 Fix: kein bfill() — stattdessen letztes Pre-Start-Signal verwenden,
    # damit der erste Handelstag nicht vom nächsten verfügbaren Signal gefüllt wird
    # (Look-Ahead-Bias-Vermeidung).
    w_daily = weights_monthly.reindex(close.index, method="ffill").shift(1)
    pre_start = weights_monthly[weights_monthly.index < close.index[0]]
    if len(pre_start) > 0:
        w_daily = w_daily.fillna(pre_start.iloc[-1])
    else:
        w_daily = w_daily.bfill()                      # Fallback: kein Pre-Start-Signal

    # Vorberechnete Renditen für den T+1-Split:
    prev_close = close.shift(1)
    ret_cc     = close.pct_change()                    # Close→Close (voller Tag)
    ret_ovn    = (open_ - prev_close) / prev_close     # Overnight: Close→Open
    ret_intra  = (close - open_) / open_               # Intraday:  Open→Close

    # Bug #4 Fix: Numpy-Arrays vor der Schleife — kein .loc[date] im Loop
    # (~5000 Iterationen: numpy-Indexierung statt pandas-Label-Lookup)
    cc_arr    = ret_cc.fillna(0.0).values
    ovn_arr   = ret_ovn.fillna(0.0).values
    intra_arr = ret_intra.fillna(0.0).values
    w_arr     = w_daily.values
    dates     = close.index

    pv        = 1.0                                    # Startwert des Portfolios
    current_w = w_arr[0].copy()                        # aktuelle (gedriftete) Gewichte
    rows      = []

    for i in range(len(dates)):
        w_target = w_arr[i]                            # Zielgewicht für diesen Tag

        # REBALANCING-ENTSCHEIDUNG: Nur handeln wenn Drift > Band
        if band == 0.0 or np.any(np.abs(w_target - current_w) > band):
            w_trade = w_target.copy()                  # → rebalancieren
        else:
            w_trade = current_w.copy()                 # → Band hält, kein Trade

        rebalanced = not np.allclose(w_trade, current_w, atol=1e-9)
        to   = float(np.sum(np.abs(w_trade - current_w)) / 2.0)   # einseit. Umschlag
        cost = to * tc * 2                             # Roundtrip-TC

        if i > 0:
            if rebalanced:
                # Bug #2 Fix: TC geometrisch zwischen Overnight und Intraday —
                # Kosten fallen bei Umschichtung am Open an, reduzieren das
                # für den Intraday-Zeitraum verfügbare Kapital.
                # PV *= (1 + r_ovn) × (1 − cost) × (1 + r_intra)
                ovn_ret   = float(np.dot(current_w, ovn_arr[i]))
                intra_ret = float(np.dot(w_trade,   intra_arr[i]))
                pv *= (1.0 + ovn_ret) * (1.0 - cost) * (1.0 + intra_ret)
            else:
                # Normaler Tag: volle Close-to-Close Rendite mit gedrifteten Gewichten
                ret = float(np.dot(current_w, cc_arr[i]))
                pv *= (1.0 + ret)                      # cost=0 (kein Rebalancing)

        rows.append({"date": dates[i], "portfolio_value": pv,
                     "a1_weight": float(w_trade[0]), "a2_weight": float(w_trade[1]),
                     "turnover": to, "tc_cost": cost})

        # Tägliche Gewichtsdrift: An Rebalancing-Tagen driften neue Gewichte (w_trade)
        # nur mit der Intraday-Rendite (Open→Close), da die Umschichtung am Open erfolgte.
        # An normalen Tagen: Full Close-to-Close Rendite (kein Split nötig).
        # Bug #3: bereits korrekt — Rebalancing-Tage nutzen intra, normale cc.
        r_drift   = intra_arr[i] if rebalanced else cc_arr[i]
        w_eod     = w_trade * (1.0 + r_drift)
        s         = w_eod.sum()
        current_w = (w_eod / s) if s > 1e-9 else w_trade.copy()

    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# §4  HILFSFUNKTIONEN  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================

# ── §4a  ROLLIERENDER Z-SCORE ─────────────────────────────────────────────
# Misst wie ungewöhnlich die aktuelle Tagesrendite im Vergleich zur
# historischen Verteilung der letzten `window` Handelstage ist.
# Z = (r_tägl − μ_historisch) / σ_historisch
#
# shift(1) → Statistiken basieren NUR auf Vergangenheitsdaten (kein Bias):
#   mu[Tag T]  = Mittelwert der Tagesrenditen von T-window bis T-1
#   sig[Tag T] = Std der Tagesrenditen von T-window bis T-1
#
# `window` ist in HANDELSTAGEN angegeben (Minimum 21 ≈ 1 Monat).
# Das Signal wird am Monatsende abgetastet → Z-Score des letzten Handelstags
# des Monats relativ zu den vorangegangenen `window` Tagen.
# Z << −3 = extremer Tageseinbruch am Monatsende → Crash-Filter F1 wird ausgelöst
def _zscore_preceding(ret, window):
    mu  = ret.shift(1).rolling(window).mean()
    sig = ret.shift(1).rolling(window).std()
    return (ret - mu) / sig.replace(0, np.nan)   # σ=0 → NaN statt ±Inf (z.B. konstanter Preis)


# ── §4b  ANALYTISCHE VOL-SKALIERUNG ──────────────────────────────────────
# Findet das Aktiengewicht w so dass das Portfolio genau die Zielvolatilität
# σ_target erreicht. Löst die 2-Asset-Varianzgleichung analytisch:
#
#   σ²_portfolio = w²σ₁² + (1-w)²σ₂² + 2w(1-w)ρσ₁σ₂  =  σ_target²
#
# Umformen → quadratische Gleichung aw² + bw + c = 0 → Mitternachtsformel.
# Vorteil: exakt, deterministisch, kein numerischer Optimierer nötig.
def _vol_scale_weight(daily_close, date, target_vol, vol_window_days):
    """Analytische Lösung der 2-Asset-Varianzgleichung (kein Optimierer nötig).

    Portfoliovarianz: V(w) = w²σ₁² + (1-w)²σ₂² + 2w(1-w)ρσ₁σ₂ = σ_target²
    Umgeformt: aw² + bw + c = 0  →  Mitternachtsformel.
    Stabiler und schneller als scipy.minimize_scalar.
    """
    hist = daily_close[daily_close.index <= date].tail(vol_window_days + 1)
    if len(hist) < vol_window_days + 1:   # +1 weil pct_change() eine Zeile kostet
        return 0.80
    ret = hist.pct_change().dropna()
    ann = np.sqrt(252)
    s1  = ret.iloc[:, 0].std() * ann
    s2  = ret.iloc[:, 1].std() * ann
    rho = float(np.clip(ret.iloc[:, 0].corr(ret.iloc[:, 1]), -1.0, 1.0))

    a = s1**2 + s2**2 - 2*rho*s1*s2
    b = 2*rho*s1*s2 - 2*s2**2
    c = s2**2 - target_vol**2

    if abs(a) < 1e-12:                    # Linearer Sonderfall
        w = -c / b if abs(b) > 1e-12 else 0.8
        return float(np.clip(w, 0.0, 1.0))

    disc = b**2 - 4*a*c
    if disc < 0:
        # Zielvolatilität außerhalb des erreichbaren Bereichs → nächste Grenze
        return 0.0 if abs(s2 - target_vol) <= abs(s1 - target_vol) else 1.0

    sq = np.sqrt(disc)
    candidates = [(-b + sq) / (2*a), (-b - sq) / (2*a)]
    valid = [w for w in candidates if 0.0 <= w <= 1.0]
    if valid:
        return float(max(valid))          # Bevorzuge höheres Aktiengewicht
    return float(np.clip(min(candidates, key=lambda x: abs(np.clip(x, 0, 1) - x)), 0.0, 1.0))


# =============================================================================
# §5  STRATEGIE A: MOMENTUM  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Die Momentum-Strategie entscheidet jeden Monat über die Allokation anhand
# einer dreistufigen Filterlogik (Priorität von oben nach unten):
#
#   F1 CRASH-SCHUTZ (höchste Priorität)
#      Wenn Z-Score < crash_thr (Standard: −3.0)
#      → 20% Aktien / 80% Gold  (defensiv)
#      Logik: Extreme Kurseinbrüche werden erkannt und abgefedert.
#
#   F2 MOMENTUM-FILTER
#      Wenn 12-2-Momentum ≤ 0  (Kurs heute unter Kurs vor 12 Monaten)
#      → 20% Aktien / 80% Gold  (defensiv)
#      Formel: mom = Preis[M-2] / Preis[M-12] − 1   (12-2-Regel nach Jegadeesh)
#      Logik: Negativer Trend → lieber Gold als fallende Aktien halten.
#
#   F3 VOL-SCALING (Normalfall)
#      Wenn weder F1 noch F2 → positiver Trend, kein Crash
#      → w% Aktien, (1-w)% Gold, wobei w so gewählt dass σ_portfolio = σ_ziel
#      Logik: Im Aufwärtstrend maximale Aktienquote, aber Vol begrenzen.
#
# WARUM 12-2 statt 12-1?
#   Die Rendite des letzten Monats (T-1) zeigt oft Short-Term-Reversal.
#   Durch Überspringen von T (aktueller Monat) und T-1 wird dieser Noise entfernt.
#   → Numerator: Preis vor 2 Monaten (T-2), Denominator: Preis vor 12 Monaten (T-12)
# =============================================================================

@st.cache_data(show_spinner="Berechne Momentum-Gewichte …")
def _momentum_weights_cached(daily_key, crash_thr, target_vol, zscore_win, vol_win_days,
                              ticker1="SPY", ticker2="GLD"):
    """Gecachte Berechnung — nur neu bei Parameteränderung.

    Momentum-Formel: shift(2)/shift(12) - 1  →  12-2-Regel
      Überspringt aktuellen Monat (T) UND letzten Monat (T-1), um Short-Term-Reversal
      aus dem Signal herauszuhalten. Standard in akademischer Literatur (Jegadeesh & Titman).

    Daten ab 2002 → kein Warmup-Problem ab 2005.
    Deep-Copy-Rückgabe → kein Cache-Mutation-Bug.
    """
    daily_close, _ = load_prices(ticker1, ticker2)          # open_ nicht benötigt
    t1, t2         = daily_close.columns[0], daily_close.columns[1]
    m_prices       = get_monthly(daily_close)
    # Signale auf ticker1-only ab 2002 berechnen (t2 wie GLD fehlt vor Nov 2004)
    t1_daily       = load_t1_prices(ticker1)                    # Tagesschlusskurse ab 2002
    ret_t1_daily   = t1_daily.pct_change()                      # tägliche Renditen
    z_daily        = _zscore_preceding(ret_t1_daily, zscore_win)# Fenster in Handelstagen
    # Abtasten am Monatsende: resample("ME").last() statt reindex(),
    # weil load_t1_prices und load_prices u.U. leicht abweichende Handelstage
    # liefern → exaktes reindex() würde dann NaN produzieren.
    z_s_full       = z_daily.resample("ME").last()              # letzter Wert je Kalendermonat
    z_s            = z_s_full.reindex(m_prices.index)           # auf m_prices-Index ausrichten
    t1_monthly     = load_t1_prices(ticker1).resample("ME").last()
    ret_t1         = t1_monthly.pct_change()                    # nur noch für mom_s benötigt
    # 12-2-Regel: skip aktuellen Monat (shift 1) UND letzten Monat (shift 2)
    mom_s          = t1_monthly.shift(2) / t1_monthly.shift(12) - 1

    rows, filters, vs_vals = [], [], []
    for date in m_prices.index:
        z   = z_s.get(date, np.nan)
        mom = mom_s.get(date, np.nan)
        if pd.isna(z) or pd.isna(mom):
            rows.append((0.80, 0.20)); filters.append("warmup"); vs_vals.append(np.nan)
        elif z < crash_thr:
            rows.append((0.20, 0.80)); filters.append("F1_crash"); vs_vals.append(np.nan)
        elif mom <= 0:
            rows.append((0.20, 0.80)); filters.append("F2_momentum"); vs_vals.append(np.nan)
        else:
            w = _vol_scale_weight(daily_close, date, target_vol, vol_win_days)
            rows.append((w, 1-w)); filters.append("F3_volscale"); vs_vals.append(w)

    idx = m_prices.index
    return (
        pd.DataFrame(rows, index=idx, columns=[t1, t2]).copy(),
        pd.Series(filters,  index=idx, name="filter").copy(),
        z_s.copy(),
        mom_s.copy(),
        pd.Series(vs_vals,  index=idx, name="volscale_w").copy(),
        z_s_full.copy(),   # vollständige Monatsreihe ab 2002 inkl. Warmup-NaN für W5b/W5c
    )


# =============================================================================
# §6  STRATEGIE B: CONTRARIAN  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Die Contrarian-Strategie nutzt Mean-Reversion: Wenn der Aktienkurs stark
# über seinem gleitenden Durchschnitt (EMA) liegt → überkauft → reduziere
# Aktienanteil. Liegt er darunter → überverkauft → erhöhe Aktienanteil.
#
# SIGNAL: EMA-basierter Z-Score
#   Z = (Preis − EMA) / rollende_Std(Preis)
#   Positives Z → Preis über EMA → reduziere Aktien
#   Negatives Z → Preis unter EMA → erhöhe Aktien
#
# ALLOKATIONSFORMEL (tanh-basiert, stetig und begrenzt):
#   w = clip(w_neutral + α · tanh(−β · Z),  0,  1)
#   w_neutral = Basisgewicht bei Z=0 (Standard: 0.80)
#   α         = maximale Abweichung vom Basisgewicht  (Standard: 0.50)
#   β         = Steilheit der Reaktion auf Z           (Standard: 0.75)
#
# WARUM tanh?
#   Begrenzt die Allokation automatisch auf [0,1] ohne Sprünge.
#   Bei extremem Z → tanh(x) → ±1 → Gewicht nähert sich 0 oder 1 an.
# =============================================================================

def _contrarian_zscore(prices, ema_win):
    """EMA-basierter Z-Score auf Preisebene.

    Dimensionsbruch-Fix: Nenner = rollende Std der Preise (gleiche Einheit wie Zähler).
    Vorher war: pct_change().std() * prices  →  Mischung aus Rendite-Vol und Preisniveau,
    was den Z-Score in Bullenmärkten systematisch stauchte.
    """
    ema = prices.ewm(span=ema_win, adjust=False).mean()
    std = prices.rolling(ema_win).std()
    return (prices - ema) / std.replace(0, np.nan)


@st.cache_data(show_spinner="Berechne Contrarian-Gewichte …")
def _contrarian_weights_cached(daily_key, w_neutral, alpha, beta, ema_win,
                                ticker1="SPY", ticker2="GLD"):
    """Deep-Copy-Rückgabe → kein Cache-Mutation-Bug."""
    daily_close, _ = load_prices(ticker1, ticker2)
    t1, t2         = daily_close.columns[0], daily_close.columns[1]
    m_prices       = get_monthly(daily_close)
    z_daily        = _contrarian_zscore(daily_close[t1], ema_win)
    rows = []
    for date in m_prices.index:
        z = z_daily.get(date, np.nan)
        w = float(np.clip(w_neutral + alpha * np.tanh(-beta * z), 0, 1)) if not pd.isna(z) else w_neutral
        rows.append((w, 1-w))
    return (
        pd.DataFrame(rows, index=m_prices.index, columns=[t1, t2]).copy(),
        z_daily.copy(),
    )


# =============================================================================
# §7  PERFORMANCE-METRIKEN  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# calc_metrics() berechnet alle zentralen Kennzahlen aus dem Backtest-Output.
#
#   CAGR      Compound Annual Growth Rate = annualisierte Gesamtrendite
#             Formel: (Endwert/Startwert)^(1/Jahre) − 1
#
#   Volatilität  Tages-Std × √252  = annualisierte Schwankungsbreite
#
#   Sharpe-Ratio  = (μ_tägl × 252) / (σ_tägl × √252)  =  μ_tägl × √252 / σ_tägl
#             Implementierung: arithmetische Annualisierung, r_f = 0 (vereinfacht).
#             HINWEIS: r_f = 0 ist in akademischen Vergleichsstudien üblich, da alle
#             Strategien gleichermaßen betroffen sind (relativer Vergleich korrekt).
#             Absoluter Sharpe-Wert ist ohne r_f leicht überschätzt — muss in der
#             Bachelorarbeit explizit erwähnt werden.
#             Abweichung von CAGR-basierter Formel: μ_arith × 252 ≈ CAGR + ½σ²
#             (Jensen's Inequality) — bei niedrigen Renditen/Vols vernachlässigbar.
#
#   Sortino-Ratio  = (μ_tägl × 252) / DD_ann,  wobei DD die Downside-Deviation ist.
#             DD = √(1/N · Σ min(r_i, 0)²) × √252   (Sortino & Price 1994)
#             Nutzt ALLE Beobachtungen im Nenner (nicht nur negative Tage).
#             Bestraft nur Verluste, nicht Aufwärtsvolatilität. Ebenfalls r_f = 0.
#
#   Max Drawdown  = größter Peak-to-Trough-Verlust im gesamten Zeitraum
#             = min((PV_t / max(PV_1..PV_t)) − 1)
#
#   DD Dauer      = längste Underwater-Periode (Drawdown Duration)
#             = maximale Anzahl aufeinanderfolgender Handelstage unter dem
#               vorherigen Portfolio-Hoch. Angabe in Monaten (÷ 21 HT).
#             Ergänzt Max-DD: zwei Strategien können gleichen Max-DD haben,
#             aber sehr unterschiedlich lange brauchen um sich zu erholen.
#
#   Schiefe (Skew) = dritter zentraler Moment der täglichen Renditeverteilung.
#             Negative Schiefe → häufiger extreme Verluste als Gewinne (Tail-Risk).
#             pandas .skew() liefert den unbiased Estimator (Fisher-Pearson).
#
#   Kurtosis       = Excess-Kurtosis (viertes zentrales Moment − 3).
#             Werte > 0 → schwerere Tails als Normalverteilung ("fat tails").
#             pandas .kurtosis() liefert Fisher's Excess-Kurtosis (Normal ≈ 0).
#
#   Worst Month    = schlechteste monatliche Rendite im gesamten Zeitraum.
#             Berechnet aus PV-Reihe: pv.resample("ME").last().pct_change().min()
#
#   Calmar-Ratio  = CAGR / |Max Drawdown|   → Rendite pro Drawdown-Risiko
#
#   Turnover p.a. = durchschnittlicher jährlicher Portfolioumschlag (%)
#
#   Kosten-Drag p.a. = CAGR_brutto − CAGR_netto
#             Misst die tatsächliche Renditeeinbuße durch TC inkl. Compounding.
#             Brutto-PV = Netto-PV / Π(1 − cost_t)  →  eigene CAGR berechnen.
# =============================================================================

def calc_metrics(port, label=""):
    pv  = port["portfolio_value"]
    r   = pv.pct_change().dropna()
    ny  = len(pv) / 252
    cagr    = (pv.iloc[-1] / pv.iloc[0]) ** (1/ny) - 1
    vol     = r.std() * np.sqrt(252)
    sharpe  = r.mean()*252/vol if vol > 0 else np.nan
    down    = np.sqrt((np.minimum(r, 0)**2).mean()) * np.sqrt(252)   # Sortino & Price 1994
    sortino = r.mean()*252/down if down > 0 else np.nan
    maxdd   = ((pv / pv.cummax()) - 1).min()
    calmar  = cagr / abs(maxdd) if maxdd != 0 else np.nan
    # Längste Underwater-Periode (Drawdown Duration):
    # Wie viele aufeinanderfolgende Handelstage war das Portfolio unter seinem
    # vorherigen Hoch? Cumsum-Trick: Zähler wird beim Auftauchen über Peak genullt.
    _uw     = pv < pv.cummax()                                     # True = unter Peak
    _cum    = _uw.cumsum()
    _reset  = _cum - _cum.where(~_uw).ffill().fillna(0)
    max_uw_days = int(_reset.max()) if _uw.any() else 0
    dd_dur  = int(round(max_uw_days / 21.0))                        # ganzzahlige Monate → kein -0-Formatierungsbug
    ann_to  = port["turnover"].resample("ME").sum().mean() * 12

    # Kosten-Drag: CAGR_brutto − CAGR_netto (berücksichtigt Zinseszinseffekt der TC)
    # pv_net = prod((1+r_t)*(1-c_t)), pv_gross = prod(1+r_t) = pv_net / prod(1-c_t)
    pv_gross = float(pv.iloc[-1]) * np.exp(float(-np.log1p(-port["tc_cost"]).sum()))
    drag_pa  = (pv_gross / float(pv.iloc[0])) ** (1/ny) - 1 - cagr
    skew    = r.skew()
    kurt    = r.kurtosis()

    # Worst Month berechnen (Monatliche Renditen aus PV ableiten)
    monthly_r    = pv.resample("ME").last().pct_change().dropna()
    worst_month  = monthly_r.min()

    return {
        "label":       label,
        "CAGR":        cagr,
        "Vol":         vol,
        "Sharpe":      sharpe,
        "Sortino":     sortino,
        "Schiefe":     skew,
        "Kurtosis":    kurt,
        "Max DD":      maxdd,
        "Worst Month": worst_month,
        "DD Dauer":    dd_dur,
        "Calmar":      calmar,
        "Turnover":    ann_to,
        "Kosten-Drag": drag_pa,
    }


# =============================================================================
# §7b  BACKTEST-REPORT-ANALYZER  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Modulare Klasse zur Erzeugung strukturierter Report-DataFrames.
# Alle KPIs folgen exakten methodischen Vorgaben (252 HT p.a.):
#
#   CAGR         geometrisch:  (End/Start)^(1/J) − 1
#   Sharpe       arithmetisch, rf = 0:  μ_d·252 / (σ_d·√252)
#   Sortino      Full-Series DD (Sortino & Price 1994):
#                μ_d·252 / (√(mean(min(r,0)²))·√252)
#   Kosten-Drag  CAGR_brutto − CAGR_netto  (Compounding berücksichtigt)
#   Skew/Kurt    Fisher-Pearson via pandas .skew() / .kurtosis()
#
# Methoden:
#   performance_df()                   → KPI-Tabelle (Index=Metrik, Spalten=Strategie)
#   regime_df()                        → Regime-Analyse (Return & MaxDD pro Krise)
#   sensitivity_df(name, vals, run_fn) → Generische Parametervariation
#   sensitivity_tc_df(vals, run_fn)    → TC-Variation für beide Strategien
#   to_markdown(...)                   → Vollständiger Markdown-Bericht
# =============================================================================

class BacktestReportAnalyzer:
    """Modularer Report-Generator für Backtest-Ergebnisse.

    Nutzt calc_metrics() als Single Source of Truth für alle KPIs.
    Erzeugt strukturierte DataFrames und einen vollständigen Markdown-Export.
    """

    REGIMES = [
        ("GFC 2007–09",     "2007-10-01", "2009-03-31"),
        ("COVID 2020",      "2020-02-01", "2020-04-30"),
        ("Bärenmarkt 2022", "2022-01-01", "2022-12-31"),
    ]

    # Metrik-Definitionen: (dict-key, Anzeigename, Format, Multiplikator, Suffix)
    _KPI_DEFS = [
        ("CAGR",        "CAGR",             ".2f", 100, " %"),
        ("Vol",         "Volatilität",      ".2f", 100, " %"),
        ("Sharpe",      "Sharpe Ratio",     ".3f",   1, ""),
        ("Sortino",     "Sortino Ratio",    ".3f",   1, ""),
        ("Schiefe",     "Schiefe (Skew)",   ".2f",   1, ""),
        ("Kurtosis",    "Kurtosis",         ".2f",   1, ""),
        ("Max DD",      "Max Drawdown",     ".2f", 100, " %"),
        ("Worst Month", "Worst Month",      ".1f", 100, " %"),
        ("DD Dauer",    "DD Dauer",         ".0f",   1, " Mon."),
        ("Calmar",      "Calmar Ratio",     ".3f",   1, ""),
        ("Turnover",    "Turnover p.a.",    ".1f", 100, " %"),
        ("Kosten-Drag", "Kosten-Drag p.a.", ".3f", 100, " %"),
    ]

    def __init__(self, ports: dict):
        """
        Parameters
        ----------
        ports : dict
            {"Benchmark": bt_df, "Momentum": bt_df, "Contrarian": bt_df}
            Jeder bt_df ist das Ergebnis von _backtest() mit Spalten:
            portfolio_value, a1_weight, a2_weight, turnover, tc_cost
        """
        self.ports = ports
        self._m = {name: calc_metrics(port, name) for name, port in ports.items()}

    # ── DataFrame-Generatoren ────────────────────────────────────────────────

    def performance_df(self) -> pd.DataFrame:
        """KPI-Tabelle: Index = Metrik-Anzeigename, Spalten = Strategien.

        Werte sind fertig formatierte Strings (z. B. '8.52 %', '0.543').
        Für numerischen Zugriff: self._m[name][key].
        """
        data = {}
        for name in self.ports:
            m = self._m[name]
            label = m.get("label", name)
            vals = []
            for key, _, fmt, mult, sfx in self._KPI_DEFS:
                v = m[key] * mult
                vals.append(f"{v:{fmt}}{sfx}")
            data[label] = vals
        idx = [display for _, display, *_ in self._KPI_DEFS]
        return pd.DataFrame(data, index=idx)

    def regime_df(self) -> pd.DataFrame:
        """Regime-Analyse: Return & Max DD pro Strategie und Krisenperiode."""
        rows = []
        for rname, rs, re in self.REGIMES:
            for pname, port in self.ports.items():
                pv = port["portfolio_value"]
                sl = pv[(pv.index >= rs) & (pv.index <= re)]
                if len(sl) < 2:
                    continue
                ret = (sl.iloc[-1] / sl.iloc[0] - 1) * 100
                dd  = float(((sl / sl.cummax()) - 1).min()) * 100
                rows.append({"Regime": rname, "Strategie": pname,
                             "Return %": round(ret, 1), "Max DD %": round(dd, 1)})
        return pd.DataFrame(rows)

    def sensitivity_df(self, param_name: str, values: list, run_fn) -> pd.DataFrame:
        """Sensitivitätsanalyse für einen Parameter.

        Parameters
        ----------
        param_name : str   Spaltenname (z. B. 'β', 'Threshold')
        values     : list  Parameter-Werte zum Iterieren
        run_fn     : callable(value) → backtest_df
        """
        rows = []
        for v in values:
            port = run_fn(v)
            m = calc_metrics(port)
            rows.append({param_name: v,
                         "CAGR %": round(m["CAGR"] * 100, 2),
                         "Sharpe": round(m["Sharpe"], 3),
                         "Max DD %": round(m["Max DD"] * 100, 2),
                         "Turnover %": round(m["Turnover"] * 100, 1)})
        return pd.DataFrame(rows)

    def sensitivity_tc_df(self, tc_values: list, run_triple_fn) -> pd.DataFrame:
        """TC-Sensitivität für Benchmark, Momentum UND Contrarian.

        Parameters
        ----------
        run_triple_fn : callable(tc_bps) → (backtest_bm, backtest_mom, backtest_con)
        """
        rows = []
        for c in tc_values:
            p_b, p_m, p_c = run_triple_fn(c)
            m_b, m_m, m_c = calc_metrics(p_b), calc_metrics(p_m), calc_metrics(p_c)
            rows.append({"TC (bps)":     c,
                         "BM CAGR %":    round(m_b["CAGR"] * 100, 2),
                         "Mom CAGR %":   round(m_m["CAGR"] * 100, 2),
                         "Mom α %":      round((m_m["CAGR"] - m_b["CAGR"]) * 100, 2),
                         "Con CAGR %":   round(m_c["CAGR"] * 100, 2),
                         "Con α %":      round((m_c["CAGR"] - m_b["CAGR"]) * 100, 2),
                         "BM Sharpe":    round(m_b["Sharpe"], 3),
                         "Mom Sharpe":   round(m_m["Sharpe"], 3),
                         "Con Sharpe":   round(m_c["Sharpe"], 3)})
        return pd.DataFrame(rows)

    def sensitivity_2d_df(self, row_name: str, row_vals: list,
                          col_name: str, col_vals: list,
                          run_fn, metric: str = "CAGR",
                          base_row=None, base_col=None) -> pd.DataFrame:
        """2D-Sensitivitätsmatrix: zwei Parameter gleichzeitig variieren.

        Parameters
        ----------
        row_name / col_name : str       Parameternamen (z. B. 'α', 'β')
        row_vals / col_vals : list      Werte für Zeilen / Spalten
        run_fn              : callable(row_val, col_val) → backtest_df
        metric              : str       'CAGR' oder 'Sharpe'
        base_row / base_col : optional  Basiswerte zum Markieren (★)
        """
        mult = 100 if metric == "CAGR" else 1
        fmt  = ".2f" if metric == "CAGR" else ".3f"
        sfx  = " %" if metric == "CAGR" else ""
        data = {}
        for cv in col_vals:
            col_label = f"{col_name}={cv}"
            if base_col is not None and cv == base_col:
                col_label += " ★"
            vals = []
            for rv in row_vals:
                port = run_fn(rv, cv)
                m = calc_metrics(port)
                v = m[metric] * mult
                cell = f"{v:{fmt}}{sfx}"
                vals.append(cell)
            data[col_label] = vals
        idx = [f"{row_name}={rv}" + (" ★" if base_row is not None and rv == base_row else "")
               for rv in row_vals]
        return pd.DataFrame(data, index=idx)

    # ── Hilfsfunktionen ───────────────────────────────────────────────────────

    @staticmethod
    def _interpolate_breakeven(df: pd.DataFrame, x_col: str, y_col: str):
        """Findet per linearer Interpolation den x-Wert, bei dem y den Nulldurchgang hat.

        Gibt None zurück, wenn kein Vorzeichenwechsel in den Daten vorliegt.
        """
        xs = df[x_col].values
        ys = df[y_col].values
        for i in range(len(ys) - 1):
            if ys[i] >= 0 > ys[i + 1] or ys[i] <= 0 < ys[i + 1]:
                # Linearer Nulldurchgang zwischen i und i+1
                return float(xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
        return None

    @staticmethod
    def _df_to_md(df: pd.DataFrame, index: bool = True) -> list:
        """Konvertiert DataFrame in Markdown-Tabellenzeilen (list[str]).

        index=True  → erste Spalte ist der DataFrame-Index
        index=False → nur die Spalten, kein Index
        """
        def _cell(v):
            return "—" if pd.isna(v) else str(v)

        lines = []
        if index:
            cols = list(df.columns)
            header = "| Metrik" + " " * max(1, 18 - 6) + "| " + " | ".join(f"{c}" for c in cols) + " |"
            sep = "|" + "-" * 20 + "|" + "|".join("-" * 15 for _ in cols) + "|"
            lines += [header, sep]
            for idx_val, row in df.iterrows():
                cells = " | ".join(_cell(row[c]) for c in cols)
                lines.append(f"| {str(idx_val):<18} | {cells} |")
        else:
            cols = list(df.columns)
            header = "| " + " | ".join(str(c) for c in cols) + " |"
            sep = "|" + "|".join("---" for _ in cols) + "|"
            lines += [header, sep]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
        return lines

    # ── Vollständiger Markdown-Bericht ───────────────────────────────────────

    def to_markdown(self, params: dict,
                    filter_s=None, crash_thr: float = -3.0,
                    sensitivity_dfs: dict = None,
                    monthly_df=None) -> str:
        """Generiert den vollständigen Markdown-Bericht.

        Parameters
        ----------
        params           : dict  Parameter-Name → Wert (für §1)
        filter_s         : Series  Momentum-Filter-Zeitreihe (für §4)
        crash_thr        : float   Crash-Threshold (für §4 Label)
        sensitivity_dfs  : dict    Titel → DataFrame (für §5)
        monthly_df       : DataFrame  Monatstabelle (für §6)
        """
        L = []

        # ── §1 Parameter ─────────────────────────────────────────────────────
        t1 = params.get("Asset 1 (Aktien)", "SPY")
        t2 = params.get("Asset 2 (Hedge)", "GLD")
        L += [f"# Backtest-Bericht: Momentum vs. Contrarian ({t1} + {t2}, 2005–2025)",
              "", "## 1. Parameter", "",
              "| Parameter                   | Wert            |",
              "|-----------------------------|-----------------|"]
        for k, v in params.items():
            L.append(f"| {k:<27} | {str(v):<15} |")
        L.append("")

        # ── §2 Performance-Metriken ──────────────────────────────────────────
        L += ["## 2. Performance-Metriken (annualisiert, 2005–2025)", ""]
        L += self._df_to_md(self.performance_df(), index=True)
        L.append("")

        # ── §3 Regime-Analyse ────────────────────────────────────────────────
        L += ["## 3. Regime-Analyse", ""]
        rdf = self.regime_df()
        for rname in dict.fromkeys(rdf["Regime"]):          # Reihenfolge beibehalten
            rsub = rdf[rdf["Regime"] == rname]
            regime_dates = [(r, s, e) for r, s, e in self.REGIMES if r == rname]
            rs, re = regime_dates[0][1], regime_dates[0][2] if regime_dates else ("", "")
            L += [f"### {rname} ({rs} – {re})", "",
                  "| Strategie  | Return   | Max DD   |",
                  "|------------|----------|----------|"]
            for _, row in rsub.iterrows():
                L.append(f"| {row['Strategie']:<10} | {row['Return %']:+.1f} %  | {row['Max DD %']:.1f} %  |")
            L.append("")

        # ── §4 Filter-Statistik ──────────────────────────────────────────────
        if filter_s is not None:
            fs = filter_s["2005-01-01":]
            n_total = len(fs)
            n_f1 = int((fs == "F1_crash").sum())
            n_f2 = int((fs == "F2_momentum").sum())
            n_f3 = int((fs == "F3_volscale").sum())
            L += ["## 4. Filter-Statistik Momentum", "",
                  "| Filter                       | Monate | Anteil   |",
                  "|------------------------------|--------|----------|",
                  f"| F1 Crash-Schutz (Z < {crash_thr})  | {n_f1:<6} | {n_f1/n_total*100:.1f} %  |",
                  f"| F2 Neg. Momentum             | {n_f2:<6} | {n_f2/n_total*100:.1f} %  |",
                  f"| F3 Volatility Scaling        | {n_f3:<6} | {n_f3/n_total*100:.1f} %  |",
                  ""]

        # ── §5 Sensitivitätsanalyse ──────────────────────────────────────────
        if sensitivity_dfs:
            L += ["## 5. Sensitivitätsanalyse", ""]
            L.append("★ = Basisparametrierung")
            L.append("")
            for title, sdf in sensitivity_dfs.items():
                L += [f"### {title}", ""]
                # 2D-Matrizen haben benannten Index, 1D haben RangeIndex
                is_matrix = not isinstance(sdf.index, pd.RangeIndex)
                L += self._df_to_md(sdf, index=is_matrix)
                L.append("")

                # Break-Even-Analyse für TC-Variation
                if "TC (bps)" in sdf.columns and "Mom α %" in sdf.columns:
                    L += ["### TC Break-Even-Analyse", "",
                          "Ab welchem TC-Niveau (bps) verliert eine Strategie ihren "
                          "Brutto-Vorteil gegenüber der Benchmark (α = 0)?", ""]
                    for strat, col in [("Momentum", "Mom α %"), ("Contrarian", "Con α %")]:
                        be = self._interpolate_breakeven(sdf, "TC (bps)", col)
                        if be is not None:
                            L.append(f"- **{strat}**: Break-Even bei **{be:.1f} bps** "
                                     f"(α > 0 unter {be:.0f} bps, α < 0 darüber)")
                        elif (sdf[col] >= 0).all():
                            L.append(f"- **{strat}**: α bleibt im gesamten TC-Bereich positiv")
                        else:
                            L.append(f"- **{strat}**: α ist im gesamten TC-Bereich negativ "
                                     "(kein Break-Even)")
                    L.append("")

        # ── §6 Monatstabelle ─────────────────────────────────────────────────
        if monthly_df is not None:
            L += ["## 6. Monatstabelle (vollständig)", ""]
            L += self._df_to_md(monthly_df.reset_index(), index=False)

        return "\n".join(L)


# =============================================================================
# §8  PLOTLY-CHARTS  ✦ VISUALISIERUNG — für Bachelorarbeit weniger kritisch
# =============================================================================
# Alle Chart-Funktionen nehmen fertig berechnete DataFrames entgegen und geben
# Plotly-Figure-Objekte zurück. Die Logik dahinter (was gezeigt wird) ist
# wichtig, aber die genaue Plotly-Syntax muss nicht verstanden werden.
#
# chart_equity()           → Equity Curves (Portfoliowert über Zeit)
# chart_drawdown()         → Drawdown-Verlauf (Verlust vom letzten Hoch)
# chart_allocation()       → Gewichtsverlauf aller Strategien
# chart_rolling_sharpe()   → Rollierendes Sharpe Ratio (1–3 Jahre)
# chart_filter_history()   → Balkendiagramm: Welcher Filter war wann aktiv
# chart_contrarian_z()     → Contrarian Z-Score + Allokation
# chart_asset_performance() → Einzelrenditen SPY & GLD normiert auf 1
# chart_return_distribution()→ Histogramm + KDE der täglichen Renditeverteilungen
# =============================================================================

def _add_regimes(fig, row=1, col=1):
    for rs, re, label, color in REGIMES:
        fig.add_vrect(
            x0=rs, x1=re,
            fillcolor=color, line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=10,
            row=row, col=col,
        )


def chart_equity(ports):
    fig = go.Figure()
    for name, p in ports.items():
        pv = p["portfolio_value"]
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv,
            name=name, line=dict(color=COLORS[name], width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}}<extra></extra>",
        ))
    _add_regimes(fig)
    fig.update_layout(
        title="Equity Curves (Start = 1 EUR)",
        yaxis_title="Portfoliowert",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_drawdown(ports):
    fig = go.Figure()
    for name, p in ports.items():
        pv = p["portfolio_value"]
        dd = ((pv / pv.cummax()) - 1) * 100
        # Hex -> rgba für fillcolor (Plotly akzeptiert kein '#rrggbbaa')
        hex_to_rgba = {
            "#555555": "rgba(85,85,85,0.15)",
            "#1f77b4": "rgba(31,119,180,0.15)",
            "#d62728": "rgba(214,39,40,0.15)",
        }
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=name, line=dict(color=COLORS[name], width=1.5),
            fill="tozeroy", fillcolor=hex_to_rgba.get(COLORS[name], "rgba(128,128,128,0.15)"),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d.%m.%Y}}<br>DD: %{{y:.1f}}%<extra></extra>",
        ))
    _add_regimes(fig)
    fig.update_layout(
        title="Drawdown-Verlauf (%)",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_allocation(w_dict):
    fig = make_subplots(
        rows=len(w_dict), cols=1,
        shared_xaxes=True,
        subplot_titles=list(w_dict.keys()),
        vertical_spacing=0.08,
    )
    for i, (name, w) in enumerate(w_dict.items(), start=1):
        t1, t2 = w.columns[0], w.columns[1]
        fig.add_trace(go.Scatter(
            x=w.index, y=w[t1],
            name=t1 if i == 1 else None,
            stackgroup="one",
            line=dict(width=0), fillcolor="rgba(33,150,243,0.75)",
            showlegend=(i == 1),
            hovertemplate=f"{t1}: %{{y:.1%}}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=w.index, y=w[t2],
            name=t2 if i == 1 else None,
            stackgroup="one",
            line=dict(width=0), fillcolor="rgba(255,193,7,0.75)",
            showlegend=(i == 1),
            hovertemplate=f"{t2}: %{{y:.1%}}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=i, col=1)
    fig.update_layout(
        title="Allokationshistorie",
        height=200 * len(w_dict) + 80,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig


def chart_rolling_sharpe(ports, window=252):
    fig = go.Figure()
    for name, p in ports.items():
        r  = p["portfolio_value"].pct_change().dropna()
        rs = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs,
            name=name, line=dict(color=COLORS[name], width=1.8),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d.%m.%Y}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    _add_regimes(fig)
    fig.update_layout(
        title=f"Rollierendes Sharpe Ratio ({window//252}-Jahr-Fenster)",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_filter_history(filter_s):
    """Balkendiagramm: Momentum-Filter-Aktivierungen pro Jahr (Handelszeitraum 2005+).

    Warmup-Einträge werden aus dem Chart ausgeblendet — sie gehören zur Vorlaufphase
    (vor 2005) und sollen im Handelszeitraum nie auftreten.
    """
    # Nur Handelszeitraum, Warmup-Artefakte ausblenden
    df = filter_s["2005-01-01":].to_frame("filter")
    df = df[df["filter"] != "warmup"]
    df["year"] = df.index.year
    counts = df.groupby(["year", "filter"]).size().unstack(fill_value=0)
    fig = go.Figure()
    color_map = {
        "F1_crash":    "#d62728",
        "F2_momentum": "#ff7f0e",
        "F3_volscale": "#1f77b4",
    }
    for f in ["F1_crash", "F2_momentum", "F3_volscale"]:
        if f in counts.columns:
            fig.add_trace(go.Bar(
                x=counts.index, y=counts[f],
                name=f.replace("_", " "),
                marker_color=color_map.get(f, "#888"),
            ))
    fig.update_layout(
        barmode="stack",
        title="Momentum-Filter-Aktivierungen pro Jahr",
        yaxis_title="Monate",
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_contrarian_z(z_daily, w_con_daily, ticker1="SPY"):
    """Z-Score Verlauf der Contrarian-Strategie mit übergelagerter Allokation."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["Contrarian Z-Score (täglich)", f"{ticker1}-Gewicht (Contrarian)"],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=z_daily.index, y=z_daily,
        line=dict(color="#d62728", width=1),
        name="Z-Score",
        hovertemplate="%{x|%d.%m.%Y}: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=1)
    for v in [-2, 2]:
        fig.add_hline(y=v, line_dash="dash", line_color="#aaa", line_width=0.8, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=w_con_daily.index, y=w_con_daily,
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)",
        line=dict(color="#d62728", width=1.2),
        name=f"{ticker1}-Gewicht",
        hovertemplate="%{x|%d.%m.%Y}: %{y:.1%}<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(
        title="Contrarian: Z-Score & Allokation",
        height=420, hovermode="x unified",
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_asset_performance(daily, ticker1, ticker2):
    """Gesamtrendite der einzelnen Assets, normiert auf 1 ab 2005-01-01."""
    fig = go.Figure()
    asset_styles = {ticker1: ("#2196F3", "solid"), ticker2: ("#FFB300", "solid")}
    for t, (color, dash) in asset_styles.items():
        price = daily[t]["2005-01-01":]
        norm  = price / price.iloc[0]
        cagr  = (norm.iloc[-1] ** (1 / (len(norm) / 252)) - 1) * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm,
            name=f"{t}  (CAGR {cagr:.1f}%)",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"<b>{t}</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}}x<extra></extra>",
        ))
    _add_regimes(fig)
    fig.update_layout(
        title=f"Asset-Einzelrenditen: {ticker1} & {ticker2}  (normiert auf 1, 2005–2025)",
        yaxis_title="Normierter Kurs",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_asset_vol_return(daily, ticker1, ticker2, vol_window=252):
    """Zwei-Panel-Chart: oben normierte Kursrenditen, unten rollende annualisierte Volatilität.

    Rollende Vol = Tages-Std(Renditen) × √252 über vol_window Handelstage.
    Zeigt klar wann ein Asset risikoreich ist und ob Diversifikation sinnvoll ist.
    """
    colors = {ticker1: "#2196F3", ticker2: "#FFB300"}
    daily_s = daily["2005-01-01":]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=[
            f"Normierter Kurs (Start = 1,  CAGR in Legende)",
            f"Rollende annualisierte Volatilität  ({vol_window}-Tage-Fenster)",
        ],
    )

    # ── Panel 1: normierte Kursrenditen ──────────────────────────────────────
    for t, color in colors.items():
        price = daily_s[t].dropna()
        norm  = price / price.iloc[0]
        ny    = len(norm) / 252
        cagr  = (norm.iloc[-1] ** (1 / ny) - 1) * 100 if ny > 0 else 0
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm,
            name=f"{t}  CAGR {cagr:.1f}%",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{t}</b><br>%{{x|%d.%m.%Y}}<br>Kurs: %{{y:.3f}}x<extra></extra>",
            legendgroup=t,
        ), row=1, col=1)

    # ── Panel 2: rollende annualisierte Volatilität ───────────────────────────
    for t, color in colors.items():
        ret     = daily_s[t].pct_change().dropna()
        roll_vol = ret.rolling(vol_window).std() * np.sqrt(252) * 100   # in Prozent
        # Mittelwert der gesamten Vol als gestrichelte Linie
        mean_vol = float(roll_vol.dropna().mean())
        fig.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol,
            name=f"{t}  Ø {mean_vol:.1f}%",
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{t} Vol</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.1f}}% p.a.<extra></extra>",
            legendgroup=t,
            showlegend=False,       # schon in Panel 1 in Legende
        ), row=2, col=1)
        # Mittelwertlinie
        fig.add_hline(
            y=mean_vol, line_dash="dot", line_color=color, line_width=1,
            annotation_text=f"Ø {t} {mean_vol:.1f}%",
            annotation_font_size=10,
            annotation_position="top right",
            row=2, col=1,
        )

    # ── Korrelations-Annotation ───────────────────────────────────────────────
    ret1 = daily_s[ticker1].pct_change().dropna()
    ret2 = daily_s[ticker2].pct_change().dropna()
    rho  = float(ret1.corr(ret2))
    fig.add_annotation(
        text=f"Korrelation {ticker1}/{ticker2}: ρ = {rho:.3f}",
        xref="paper", yref="paper", x=0.01, y=0.44,
        showarrow=False, font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.75)", bordercolor="#ccc", borderwidth=1,
    )

    # ── Regime-Markierungen auf beiden Panels ────────────────────────────────
    _add_regimes(fig, row=1, col=1)
    _add_regimes(fig, row=2, col=1)

    fig.update_yaxes(title_text="Normierter Kurs", row=1, col=1)
    fig.update_yaxes(title_text="Volatilität (% p.a.)", row=2, col=1)
    fig.update_layout(
        title=f"{ticker1} & {ticker2} — Rendite & Volatilität  (2005–2025)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=620,
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def _hex_to_rgba(hex_color, alpha):
    """Konvertiert Hex-Farbe (#RRGGBB) zu rgba()-String."""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def chart_return_distribution(ports, bw_adjust=1.5):
    """KDE-Dichteplot der täglichen Renditeverteilungen mit Log-Y-Achse.

    Zwei Subplots für klaren Vergleich ohne Spaghetti-Effekt:
      Oben:  Benchmark vs. Momentum  → sichtbare Tail-Dämpfung durch Filter
      Unten: Benchmark vs. Contrarian → breitere Ränder, positive Schiefe
    Log-Y-Achse hebt die Tails hervor — essentiell für Tail-Risk-Analyse.
    Normalverteilungsreferenz (gestrichelt) in beiden Subplots.
    Skewness- und Kurtosis-Werte direkt annotiert.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Benchmark vs. Momentum", "Benchmark vs. Contrarian"],
        shared_xaxes=True,
        vertical_spacing=0.10,
    )

    # ── Tägliche Renditen berechnen ──────────────────────────────────────────
    daily_rets = {}
    for name, p in ports.items():
        daily_rets[name] = p["portfolio_value"].pct_change().dropna() * 100  # in %

    # ── Gemeinsame X-Achse und Normalverteilung ──────────────────────────────
    all_r = pd.concat(daily_rets.values())
    x_lo, x_hi = float(all_r.min()) - 0.5, float(all_r.max()) + 0.5
    x_range = np.linspace(x_lo, x_hi, 800)

    r_bm = daily_rets["Benchmark"]
    mu_bm, sig_bm = float(r_bm.mean()), float(r_bm.std())
    y_norm = np.maximum(norm.pdf(x_range, mu_bm, sig_bm), 1e-8)

    # ── KDE + Normalverteilung in beide Subplots ─────────────────────────────
    pairs = [
        (1, "Momentum"),
        (2, "Contrarian"),
    ]

    for row, strat_name in pairs:
        # Benchmark KDE (in jedem Subplot als Referenz)
        kde_bm = gaussian_kde(r_bm.values, bw_method="scott")
        kde_bm.set_bandwidth(kde_bm.factor * bw_adjust)
        y_bm   = np.maximum(kde_bm(x_range), 1e-8)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_bm,
            name="Benchmark", legendgroup="BM",
            showlegend=(row == 1),
            line=dict(color=COLORS["Benchmark"], width=2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(COLORS["Benchmark"], 0.15),
            hovertemplate="<b>Benchmark</b><br>%{x:.2f}%<br>Dichte: %{y:.2e}<extra></extra>",
        ), row=row, col=1)

        # Strategie-KDE
        r_s   = daily_rets[strat_name]
        kde_s = gaussian_kde(r_s.values, bw_method="scott")
        kde_s.set_bandwidth(kde_s.factor * bw_adjust)
        y_s   = np.maximum(kde_s(x_range), 1e-8)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_s,
            name=strat_name, legendgroup=strat_name,
            showlegend=(row == 1),
            line=dict(color=COLORS[strat_name], width=2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(COLORS[strat_name], 0.20),
            hovertemplate=f"<b>{strat_name}</b><br>%{{x:.2f}}%<br>Dichte: %{{y:.2e}}<extra></extra>",
        ), row=row, col=1)

        # Normalverteilung (gestrichelt)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm,
            name="Normalvert.", legendgroup="norm",
            showlegend=(row == 1),
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            hoverinfo="skip",
        ), row=row, col=1)

        # ── Annotations: Skewness & Kurtosis (oben links) ────────────────
        skew_bm, kurt_bm = float(r_bm.skew()), float(r_bm.kurtosis())
        skew_s,  kurt_s  = float(r_s.skew()),  float(r_s.kurtosis())
        ann_text = (f"<b>Benchmark</b>: Skew {skew_bm:+.2f}  Kurt {kurt_bm:.2f}<br>"
                    f"<b>{strat_name}</b>: Skew {skew_s:+.2f}  Kurt {kurt_s:.2f}")
        # yref auf paper-Koordinaten: Subplot 1 oben ≈ 0.98, Subplot 2 oben ≈ 0.45
        y_paper = 0.98 if row == 1 else 0.45
        fig.add_annotation(
            text=ann_text, xref="paper", yref="paper",
            x=0.01, y=y_paper,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=13, family="monospace"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
        )

    # ── Layout: Log-Y mit wissenschaftlicher Notation (10⁻², 10⁻³ …) ────────
    for row in (1, 2):
        fig.update_yaxes(
            type="log", dtick=1, row=row, col=1,
            title_text="Dichte (log)",
            exponentformat="power",
        )
    fig.update_xaxes(title_text="Tagesrendite (%)", row=2, col=1)

    fig.update_layout(
        title="Renditeverteilung (täglich) — KDE mit Log-Skala",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        height=700,
        margin=dict(l=50, r=20, t=80, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §9  MONATSTABELLE  ✦ APP-INFRASTRUKTUR — für Bachelorarbeit weniger kritisch
#  Baut eine Tabelle mit Monatsrenditen, Drawdowns und Filteraktivierungen.
#  Wichtig: cache_key erzwingt Neuberechnung bei Parameteränderung (kein Cache-Freeze).
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_monthly_table(cache_key,
                         _p_bm, _p_mom, _p_con, _w_mom_daily, _w_con_daily,
                         _filter_s, _z_mon_s, _z_con_daily):
    """Baut die Monatstabelle.
    cache_key  – wird gehasht und erzwingt Neuberechnung bei Parameteränderung.
    Argumente mit _ werden NICHT gehasht (DataFrames zu teuer zum Hashen).
    """
    rows = []
    for mp in pd.period_range("2005-01", "2025-12", freq="M"):
        ms, me = mp.start_time.normalize(), mp.end_time.normalize()
        td = _p_bm[((_p_bm.index >= ms) & (_p_bm.index <= me))]
        if len(td) < 2: continue
        s, e = td.index[0], td.index[-1]

        def mret(port): sl=port["portfolio_value"][(port.index>=ms)&(port.index<=me)]; return (sl.iloc[-1]/sl.iloc[0]-1)*100 if len(sl)>=2 else np.nan
        def mmdd(port): sl=port["portfolio_value"][(port.index>=ms)&(port.index<=me)]; return float(((sl/sl.cummax())-1).min())*100 if len(sl)>=2 else np.nan

        bm_r = mret(_p_bm); mom_r = mret(_p_mom); con_r = mret(_p_con)
        f_m  = _filter_s[(_filter_s.index>=ms)&(_filter_s.index<=me)]
        z_m  = _z_mon_s[(_z_mon_s.index>=ms)&(_z_mon_s.index<=me)].dropna()
        zc_m = _z_con_daily[(_z_con_daily.index>=ms)&(_z_con_daily.index<=me)].dropna()

        rows.append({
            "Monat":            f"{e.year}-{e.month:02d}",
            "Regime":           ("Crash" if bm_r and bm_r<-5 else "Bear" if bm_r and bm_r<-1 else "Bull" if bm_r and bm_r>3 else "Sideways"),
            "BM %":             round(bm_r, 2) if not pd.isna(bm_r) else np.nan,
            "Mom %":            round(mom_r, 2) if not pd.isna(mom_r) else np.nan,
            "α Mom %":          round(mom_r-bm_r, 2) if not pd.isna(mom_r) else np.nan,
            "Con %":            round(con_r, 2) if not pd.isna(con_r) else np.nan,
            "α Con %":          round(con_r-bm_r, 2) if not pd.isna(con_r) else np.nan,
            "Mom DD %":         round(mmdd(_p_mom), 2),
            "Con DD %":         round(mmdd(_p_con), 2),
            "Mom Filter":       f_m.value_counts().idxmax().replace("_"," ") if len(f_m)>0 else "—",
            "Mom F1":           int((f_m=="F1_crash").sum()),
            "Mom F2":           int((f_m=="F2_momentum").sum()),
            "Mom F3":           int((f_m=="F3_volscale").sum()),
            "Z Mom ⌀":         round(float(z_m.mean()), 2) if len(z_m)>0 else np.nan,
            "Mom w_A1 ⌀":      round(float(_w_mom_daily[(_w_mom_daily.index>=ms)&(_w_mom_daily.index<=me)].mean()), 2),
            "Z Con ⌀":         round(float(zc_m.mean()), 2) if len(zc_m)>0 else np.nan,
            "Z Con end":        round(float(_z_con_daily.asof(e)), 2) if e in _z_con_daily.index else np.nan,
            "Con w_A1 ⌀":      round(float(_w_con_daily[(_w_con_daily.index>=ms)&(_w_con_daily.index<=me)].mean()), 2),
            "Bester":           max([("BM",bm_r),("Mom",mom_r),("Con",con_r)], key=lambda x: x[1] if not pd.isna(x[1]) else -99)[0],
        })
    return pd.DataFrame(rows).set_index("Monat")


# =============================================================================
# §11  LOGIK-CHECKS  ★ WICHTIG FÜR BACHELORARBEIT
# =============================================================================
# Automatisierte Selbstprüfung aller Berechnungen.
# Jeder Check ist unabhängig — ein Fehler blockiert keine anderen Tests.
# Gruppiert in 6 Themenblöcke A–F:
#
#   A · Datenkonsistenz          W1–W4   (Gewichte, Portfoliowert, TC-Formel)
#   B · Signalberechnung         W5–W7   (Warmup, Momentum-Formel, Z-Score)
#   C · Backtest-Engine          W8–W13  (Renditen, Band, Vol-Scaling, T+1-Split)
#   D · Filterlogik              W14,W17–W19,W21 (Hierarchie, Vollständigkeit)
#   E · Contrarian-Strategie     W11,W15,W18 (Z-Score-Formel, tanh, Richtung)
#   F · Qualitäts-Plausibilität  W16,W20,W22 (Datenlücken, Frequenz, Alpha)
# =============================================================================

def _run_logic_checks(ctx):
    """Führt alle W1–W22 Logik-Checks durch und gibt eine Liste von Checks zurück.

    Parameter
    ---------
    ctx : dict
        Alle für die Checks benötigten Daten und Parameter (siehe main()).

    Rückgabe
    --------
    list[tuple[str, bool, str, bool]]
        Jedes Element: (name, ok, detail, warn)
        warn=True → ⚠ statt ✗ bei Nichtbestehen.
    """
    # ── Kontext entpacken ────────────────────────────────────────────────────
    ports       = ctx["ports"]
    m_mom       = ctx["m_mom"]
    m_bm        = ctx["m_bm"]
    m_con       = ctx["m_con"]
    daily_close = ctx["daily_close"]
    daily_open  = ctx["daily_open"]
    w_mom       = ctx["w_mom"]
    w_con       = ctx["w_con"]
    filter_s    = ctx["filter_s"]
    z_mon_s      = ctx["z_mon_s"]
    z_mon_s_full = ctx["z_mon_s_full"]   # vollständige Reihe ab 2002 inkl. Warmup-NaN
    mom_sig     = ctx["mom_sig"]
    z_con_d     = ctx["z_con_d"]
    vs_s        = ctx["vs_s"]
    m_prices    = ctx["m_prices"]
    tc_bps      = ctx["tc_bps"]
    band_pct    = ctx["band_pct"]
    band_bm     = ctx.get("band_bm", ctx["band_pct"])
    target_vol  = ctx["target_vol"]
    crash_thr   = ctx["crash_thr"]
    zscore_win  = ctx["zscore_win"]
    vol_win_d   = ctx["vol_win_d"]
    ema_win     = ctx["ema_win"]
    w_neutral   = ctx["w_neutral"]
    alpha       = ctx["alpha"]
    beta        = ctx["beta"]
    ticker1     = ctx["ticker1"]
    ticker2     = ctx["ticker2"]

    checks = []

    def ck(name, ok, detail="", warn=False):
        """warn=True → orangener ⚠ statt rotem ✗ bei Nichtbestehen."""
        checks.append((name, bool(ok), detail, warn))

    # =========================================================================
    # GRUPPE A — DATENKONSISTENZ  (W1–W4)
    # =========================================================================

    # ── W1 – Gewichte summieren auf 1 ────────────────────────────────────────
    for sname, port in ports.items():
        w_sum = port["a1_weight"] + port["a2_weight"]
        max_dev   = float((w_sum - 1.0).abs().max())
        n_days_ok = int(((w_sum - 1.0).abs() <= 1e-6).sum())
        n_days    = len(w_sum)
        # a) Max-Abweichung < 1e-7
        ok_a = max_dev < 1e-7
        # b) Tagesanzahl mit Abweichung > 1e-6
        ok_b = n_days_ok == n_days
        # c) Erster und letzter Tag summieren exakt auf 1
        first_sum = float(w_sum.iloc[0])
        last_sum  = float(w_sum.iloc[-1])
        ok_c = abs(first_sum - 1.0) < 1e-9 and abs(last_sum - 1.0) < 1e-9
        ck(f"W1a Gewichte summieren auf 1 — Max-Abweichung ({sname})",
           ok_a,
           f"Max-Dev: {max_dev:.2e}")
        ck(f"W1b Gewichte summieren auf 1 — kein Tag > 1e-6 ({sname})",
           ok_b,
           f"{n_days_ok}/{n_days} Tage in Toleranz")
        ck(f"W1c Gewichte summieren auf 1 — erster/letzter Tag ({sname})",
           ok_c,
           f"Erster: {first_sum:.10f}  |  Letzter: {last_sum:.10f}")

    # ── W2 – Gewichte in [0, 1] ───────────────────────────────────────────────
    for sname, port in ports.items():
        # a) Keine Gewichte außerhalb [-1e-9, 1+1e-9]
        ok_a = bool(((port["a1_weight"] >= -1e-9) & (port["a1_weight"] <= 1 + 1e-9) &
                     (port["a2_weight"] >= -1e-9) & (port["a2_weight"] <= 1 + 1e-9)).all())
        min_w = float(min(port["a1_weight"].min(), port["a2_weight"].min()))
        max_w = float(max(port["a1_weight"].max(), port["a2_weight"].max()))
        # c) Harte Grenzen 0/1 nie verletzt
        ok_c = min_w >= 0.0 and max_w <= 1.0
        ck(f"W2a Gewichte in [-1e-9, 1+1e-9] ({sname})", ok_a,
           f"Min: {min_w:.6f}  |  Max: {max_w:.6f}")
        ck(f"W2c Gewichte strikt in [0, 1] ({sname})", ok_c,
           f"Min: {min_w:.6f}  |  Max: {max_w:.6f}")

    # b) Für Momentum: mindestens 1 F3-Monat mit ≠ 0.20 (Vol-Scaling aktiv)
    fs_trade = filter_s["2005-01-01":]
    f3_dates = fs_trade[fs_trade == "F3_volscale"].index
    if len(f3_dates) > 0:
        f3_non_default = int((
            (w_mom.reindex(f3_dates)[ticker1] - 0.20).abs() > 1e-6
        ).sum())
        ck("W2b Momentum: F3-Monate mit vol-skalierten Gewichten ≠ 0.20 vorhanden",
           f3_non_default >= 1,
           f"{f3_non_default}/{len(f3_dates)} F3-Monate mit != 0.20")
    else:
        ck("W2b Momentum: F3-Monate vorhanden", False, "Keine F3-Monate gefunden")

    # ── W3 – Portfolio-Wert valide ────────────────────────────────────────────
    for sname, port in ports.items():
        pv = port["portfolio_value"]
        # a) PV > 0 immer
        ok_a = bool((pv > 0).all())
        min_pv = float(pv.min())
        # b) PV[0] == 1.0
        ok_b = abs(float(pv.iloc[0]) - 1.0) < 1e-9
        # c) Kein NaN oder Inf
        ok_c = bool(pv.notna().all() and np.isfinite(pv.values).all())
        # d) Vernünftiger Endwert (0.1 bis 100.0 nach ~20 Jahren)
        end_pv = float(pv.iloc[-1])
        ok_d = 0.1 <= end_pv <= 100.0
        ck(f"W3a Portfolio-Wert > 0 immer ({sname})", ok_a,
           f"Min: {min_pv:.4f}")
        ck(f"W3b Portfolio-Wert beginnt bei 1.0 ({sname})", ok_b,
           f"PV[0] = {float(pv.iloc[0]):.10f}")
        ck(f"W3c Kein NaN/Inf im Portfolio-Wert ({sname})", ok_c,
           f"Länge: {len(pv)}, alle Werte endlich und definiert")
        ck(f"W3d Portfolio-Endwert in plausiblem Bereich 0.1–100× ({sname})", ok_d,
           f"Endwert: {end_pv:.3f}x", warn=True)

    # ── W4 – TC-Formel (Roundtrip-Konsistenz) ────────────────────────────────
    expected_ratio = tc_bps / 10_000 * 2
    for sname, port in ports.items():
        tc_ratio = (port["tc_cost"] / port["turnover"].replace(0, np.nan)).dropna()
        # a+b) cost/turnover == tc_bps/10000*2
        ok_ab = bool((tc_ratio - expected_ratio).abs().max() < 1e-10) if len(tc_ratio) > 0 else True
        # c) Tage mit Turnover=0 haben cost=0
        zero_to_days = port[port["turnover"] < 1e-15]
        ok_c = bool((zero_to_days["tc_cost"].abs() < 1e-15).all()) if len(zero_to_days) > 0 else True
        # d) Gesamt-TC im parameterabhängig plausiblen Bereich.
        # Obergrenze = theoretisches Maximum bei monatlichem Voll-Rebalancing (to=0.5)
        # über den gesamten Zeitraum:  12/Jahr × 0.5 × tc × 2 × 22 Jahre = tc × 264
        # Beispiel: 10 bps → max 26.4%, 30 bps → max 79.2%
        total_tc  = float(port["tc_cost"].sum())
        tc_upper  = (tc_bps / 10_000) * 300      # +14% Puffer über theoret. Maximum
        ok_d = 0.0 <= total_tc <= tc_upper
        mean_ratio = float(tc_ratio.mean()) if len(tc_ratio) > 0 else float("nan")
        ck(f"W4a/b TC = Turnover × bps × 2 ({sname})",
           ok_ab,
           f"Erwartet: {expected_ratio:.6f}  |  Ø Ratio: {mean_ratio:.6f}")
        ck(f"W4c Turnover=0 → cost=0 ({sname})",
           ok_c,
           f"{len(zero_to_days)} Tage mit Turnover≈0 geprüft")
        ck(f"W4d Gesamt-TC plausibel [0, {tc_upper*100:.0f}%] ({sname})",
           ok_d,
           f"Gesamt-TC: {total_tc*100:.3f}%  |  Obergrenze: {tc_upper*100:.0f}% "
           f"({tc_bps} bps × 300 Rebalancings)", warn=True)

    # =========================================================================
    # GRUPPE B — SIGNALBERECHNUNG  (W5–W7)
    # =========================================================================

    # ── W5 – Warmup-Periode korrekt ───────────────────────────────────────────
    fs_trade_2005 = filter_s["2005-01-01":]
    warmup_in_trade = int((fs_trade_2005 == "warmup").sum())
    # a) Kein Warmup ab 2005
    ck("W5a Kein Warmup im Handelszeitraum (2005+)",
       warmup_in_trade == 0,
       f"{warmup_in_trade} Warmup-Einträge ab 2005")

    # b) Warmup-Mechanismus aktiv: z_mon_s_full enthält die vollständige Reihe
    #    ab 2002 (vor dem Handelszeitraum) und zeigt das Aufwärmen des Rolling-Fensters.
    #    Wir erwarten NaN in den ersten ~ceil(zscore_win/21) Monatseinträgen.
    _warmup_months = max(1, int(np.ceil(zscore_win / 21)))
    z_nan_early = int(z_mon_s_full.iloc[:_warmup_months].isna().sum())
    ck("W5b Warmup-Mechanismus aktiv: erste Monate in z_mon_s_full sind NaN",
       z_nan_early > 0,
       f"{z_nan_early}/{_warmup_months} NaN-Einträge in den ersten {_warmup_months} Monaten "
       f"(≈{zscore_win} Handelstage Fenster) ab {z_mon_s_full.index[0].date()}")

    # c) Kein NaN mehr in z_mon_s_full nach der Warmup-Phase
    _warmup_buf = _warmup_months + 2
    z_after_warmup = z_mon_s_full.iloc[_warmup_buf:]
    z_nan_after = int(z_after_warmup.isna().sum())
    ck("W5c Kein NaN in z_mon_s_full nach Warmup-Phase",
       z_nan_after == 0,
       f"{z_nan_after} NaN-Werte nach den ersten {_warmup_buf} Monaten")

    # ── W6 – Momentum-Formel: 12-2-Regel ─────────────────────────────────────
    ms_idx = mom_sig.index
    # Gemeinsame Hilfsfunktion: mom an einem Datum prüfen
    # Referenzpreise aus load_t1_prices (ticker1-only, ab 2002)
    t1_monthly_ref = load_t1_prices(ticker1).resample("ME").last()

    def _check_mom_date(test_date_str, offset_idx):
        """Prüft ob mom_sig an einem Datum der 12-2-Formel entspricht."""
        cand = ms_idx[ms_idx >= test_date_str]
        if len(cand) <= offset_idx:
            return None, None, None
        t_date = cand[offset_idx]
        t_pos  = ms_idx.get_loc(t_date)
        if t_pos < 12:
            return None, None, None
        d_s2  = ms_idx[t_pos - 2]
        d_s12 = ms_idx[t_pos - 12]
        if d_s2 not in t1_monthly_ref.index or d_s12 not in t1_monthly_ref.index:
            return None, None, None
        expected = float(t1_monthly_ref.loc[d_s2] / t1_monthly_ref.loc[d_s12] - 1)
        actual   = float(mom_sig.loc[t_date])
        return t_date, expected, actual

    for label, ts, ofs in [("2006", "2006-01-01", 2),
                            ("2010", "2010-01-01", 5),
                            ("2015", "2015-01-01", 3)]:
        td, exp_m, act_m = _check_mom_date(ts, ofs)
        if td is None:
            ck(f"W6 Momentum-Formel shift(2)/shift(12) — {label}", True,
               "Referenzdatum außerhalb Zeitraum — Test übersprungen")
        else:
            ck(f"W6 Momentum-Formel shift(2)/shift(12) — {label}",
               abs(act_m - exp_m) < 1e-9,
               f"{td.date()}  Erwartet: {exp_m:.6f}  |  Ist: {act_m:.6f}")

    # d) shift(1) ergibt ANDEREN Wert als shift(2) → wirklich 12-2 und nicht 12-1
    cand_w6d = ms_idx[ms_idx >= "2008-01-01"]
    if len(cand_w6d) > 2:
        t_date_d = cand_w6d[2]
        t_pos_d  = ms_idx.get_loc(t_date_d)
        if t_pos_d >= 12:
            d_s1  = ms_idx[t_pos_d - 1]
            d_s12 = ms_idx[t_pos_d - 12]
            if d_s1 in t1_monthly_ref.index and d_s12 in t1_monthly_ref.index:
                mom_shift1 = float(t1_monthly_ref.loc[d_s1] / t1_monthly_ref.loc[d_s12] - 1)
                mom_shift2 = float(mom_sig.loc[t_date_d])
                ck("W6d Momentum shift(2) ≠ shift(1) — 12-2 nicht 12-1",
                   abs(mom_shift1 - mom_shift2) > 1e-9,
                   f"{t_date_d.date()} shift(2)={mom_shift2:.6f}  shift(1)={mom_shift1:.6f}")
            else:
                ck("W6d Momentum shift(2) ≠ shift(1)", True, "Referenzdaten nicht verfügbar — übersprungen")
        else:
            ck("W6d Momentum shift(2) ≠ shift(1)", True, "Zu wenig Vorlauf — übersprungen")
    else:
        ck("W6d Momentum shift(2) ≠ shift(1)", True, "Zu wenig Monate — übersprungen")

    # ── W7 – Z-Score Statistiken (Momentum) ──────────────────────────────────
    z_trade = z_mon_s["2005-01-01":]
    # a) Kein NaN ab 2005
    z_nan = int(z_trade.isna().sum())
    ck("W7a Momentum Z-Score ab 2005: kein NaN",
       z_nan == 0,
       f"{z_nan} NaN-Werte" if z_nan > 0 else "Alle Werte berechnet")

    z_trade_valid = z_trade.dropna()
    if len(z_trade_valid) > 0:
        # b) Alle Werte in (-7, +7)
        z_max_abs = float(z_trade_valid.abs().max())
        ck("W7b Momentum Z-Score abs. Max < 7",
           z_max_abs < 7.0,
           f"Max |Z|: {z_max_abs:.3f}")

        # c) Mittelwert ≈ 0 (grob zentriert)
        z_mean = float(z_trade_valid.mean())
        ck("W7c Momentum Z-Score Mittelwert ≈ 0 (|μ| < 1)",
           abs(z_mean) < 1.0,
           f"μ = {z_mean:.4f}")

        # d) Mindestens ein Monat mit Z < crash_thr (Crash-Filter ausgelöst)
        n_crash = int((z_trade_valid < crash_thr).sum())
        ck(f"W7d Mindestens 1 Monat mit Z < {crash_thr} (Crash-Filter je ausgelöst)",
           n_crash >= 1,
           f"{n_crash} Monate mit Z < {crash_thr}")

        # e) Mindestens ein Monat mit Z > +1.5 (beide Seiten der Verteilung)
        n_high = int((z_trade_valid > 1.5).sum())
        ck("W7e Mindestens 1 Monat mit Z > +1.5 (positive Seite beobachtet)",
           n_high >= 1,
           f"{n_high} Monate mit Z > 1.5")

    # =========================================================================
    # GRUPPE C — BACKTEST-ENGINE  (W8–W13)
    # =========================================================================

    # ── W8 – Tagesrenditen plausibel ─────────────────────────────────────────
    for sname, port in ports.items():
        r = port["portfolio_value"].pct_change().dropna()
        # a) |r| < 20%
        extreme = int((r.abs() > 0.20).sum())
        ck(f"W8a Tagesrenditen |r| < 20% ({sname})",
           extreme == 0,
           f"Max: {r.abs().max()*100:.2f}%"
           + (f"  |  {extreme} Ausreißer" if extreme > 0 else ""))

        # b) Kein NaN oder Inf
        ok_nan = bool(r.notna().all() and np.isfinite(r.values).all())
        ck(f"W8b Keine NaN/Inf in Renditereihe ({sname})",
           ok_nan,
           f"Länge: {len(r)}, alle Werte definiert")

        # c) Top-5 Extremtage für Benchmark: in Krisenperioden 2008–2009 oder 2020
        if sname == "Benchmark":
            top5 = r.abs().nlargest(5)
            crisis_years = {2008, 2009, 2020}
            in_crisis = int(sum(1 for d in top5.index if d.year in crisis_years))
            top3_str = ", ".join(f"{d.date()}({v*100:.1f}%)" for d, v in top5.iloc[:3].items())
            ck("W8c Benchmark: Top-5-Extremtage in Krisenperioden (2008/09 oder 2020)",
               in_crisis >= 3,
               f"In Krise: {in_crisis}/5  |  Top 3: {top3_str}", warn=True)

    # ── W9 – Rebalancing-Band reduziert Turnover ──────────────────────────────
    to_with = ports["Momentum"]["turnover"].sum()
    if band_pct == 0.0:
        ck("W9 Rebalancing-Band 0% — Vergleich übersprungen",
           True, "Band=0 → kein Referenzpunkt für Band-Reduktion")
    else:
        # a) Band > 0 → Turnover mit Band < ohne Band
        p_noband = _backtest(daily_close, daily_open, w_mom, tc_bps, band=0.0)
        to_none  = p_noband["turnover"].sum()
        ck(f"W9a Rebalancing-Band {band_pct*100:.0f}% reduziert Turnover vs. Band=0",
           to_with < to_none,
           f"Mit Band: {to_with:.2%}  |  Ohne Band: {to_none:.2%}  "
           f"|  Ersparnis: {(1-to_with/to_none)*100:.1f}%")

        # b) Wenn band_pct >= 0.02: Band=0.01 hat mehr Turnover als aktuelles Band
        if band_pct >= 0.02:
            p_halfband = _backtest(daily_close, daily_open, w_mom, tc_bps, band=0.01)
            to_half    = p_halfband["turnover"].sum()
            ck("W9b Monotonizität: Band=1% hat mehr Turnover als aktuelle Band-Einstellung",
               to_half >= to_with,
               f"Band=1%: {to_half:.2%}  |  Band={band_pct*100:.0f}%: {to_with:.2%}")

        # c) Band > 0: Momentum rebalanciert öfter als Band = band_pct * 2 (falls >0)
        if band_pct >= 0.01:
            p_dblband = _backtest(daily_close, daily_open, w_mom, tc_bps, band=min(band_pct * 2, 0.20))
            to_dbl    = p_dblband["turnover"].sum()
            ck(f"W9c Doppeltes Band ({min(band_pct*2,0.20)*100:.0f}%) hat weniger Turnover",
               to_dbl <= to_with,
               f"Band×2: {to_dbl:.2%}  |  Band={band_pct*100:.0f}%: {to_with:.2%}")

    # ── W10 – Analytische Vol-Skalierung ─────────────────────────────────────
    f3_idx    = fs_trade[fs_trade == "F3_volscale"].index[:20]
    vol_errs  = []
    vol_dates_tested = []
    for d in f3_idx:
        w_eq = float(w_mom.loc[d, ticker1])
        hist = daily_close[daily_close.index <= d].tail(vol_win_d + 1).pct_change().dropna()
        if len(hist) < vol_win_d:
            continue
        s1  = hist.iloc[:, 0].std() * np.sqrt(252)
        s2  = hist.iloc[:, 1].std() * np.sqrt(252)
        rho = float(np.clip(hist.iloc[:, 0].corr(hist.iloc[:, 1]), -1.0, 1.0))
        a_q = s1**2 + s2**2 - 2*rho*s1*s2
        b_q = 2*rho*s1*s2 - 2*s2**2
        c_q = s2**2 - target_vol**2
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
        pv_vol = np.sqrt(w_eq**2*s1**2 + (1-w_eq)**2*s2**2 + 2*w_eq*(1-w_eq)*s1*s2*rho)
        if not np.isnan(pv_vol):
            vol_errs.append(abs(pv_vol - target_vol))
            vol_dates_tested.append(d)

    if vol_errs:
        max_err  = max(vol_errs)
        mean_err = float(np.mean(vol_errs))
        # a) Max-Fehler < 1%
        ck("W10a Analytische Vol-Skalierung: |σ_ist − σ_ziel| < 1%",
           max_err < 0.01,
           f"Max: {max_err*100:.4f}%  Ø: {mean_err*100:.4f}%  ({len(vol_errs)} Fälle)")

        # b) Anderer target_vol → anderes Gewicht (kein degenerierter Scaler)
        if len(vol_dates_tested) > 0:
            d_test   = vol_dates_tested[0]
            w_orig   = float(w_mom.loc[d_test, ticker1])
            w_alt    = _vol_scale_weight(daily_close, d_test, target_vol * 1.5, vol_win_d)
            ck("W10b Vol-Scaler nicht degeneriert: target×1.5 → anderes Gewicht",
               abs(w_alt - w_orig) > 1e-6,
               f"w_orig={w_orig:.4f}  w_alt(σ×1.5)={w_alt:.4f}")
    else:
        ck("W10a Analytische Vol-Skalierung", True,
           "Keine erreichbaren F3-Fälle für diesen Parametersatz — übersprungen")

    # ── W11 – Contrarian Z-Score: std auf Preisebene ─────────────────────────
    # WICHTIG: Vollständige Preisreihe für Referenzrechnung verwenden,
    # damit die EMA korrekt aufgewärmt ist (EMA ist rekursiv → slice zerstört Warmup)
    _spy_full = daily_close[ticker1]
    # a) Direkter Vergleich Funktion vs. Referenz (vollständige Reihe, dann slice ab 2010)
    _z_func_full = _contrarian_zscore(_spy_full, ema_win)
    _ema_ref     = _spy_full.ewm(span=ema_win, adjust=False).mean()
    _std_ref     = _spy_full.rolling(ema_win).std().replace(0, np.nan)
    _z_ref_full  = (_spy_full - _ema_ref) / _std_ref
    # Vergleich ab 2010+ (EMA vollständig aufgewärmt wenn ema_win ≤ 252)
    _z_func_cmp  = _z_func_full["2010-01-01":].dropna()
    _z_ref_cmp   = _z_ref_full["2010-01-01":].dropna()
    _z_al, _z_pr = _z_func_cmp.align(_z_ref_cmp, join="inner")
    _w11_err     = float((_z_al - _z_pr).abs().max()) if len(_z_al) > 0 else float("nan")
    ck("W11a Contrarian Z-Score: Formel stimmt mit Referenz überein (Preisebene, 2010+)",
       float(_w11_err) < 1e-9,
       f"Max-Abweichung: {_w11_err:.2e}  ({len(_z_al)} gemeinsame Tage)")

    # b) Z-Score ist NaN während EMA-Warmup (erste ema_win Handelstage)
    z_early = _z_func_full.iloc[:ema_win]
    # Mindestens einige NaN in der Warmup-Phase
    n_nan_early = int(z_early.isna().sum())
    ck("W11b Z-Score hat NaN während EMA-Warmup",
       n_nan_early > 0,
       f"{n_nan_early}/{ema_win} NaN-Werte in den ersten {ema_win} Handelstagen")

    # c) Z-Score kein NaN nach Warmup (ab 2010+)
    z_2010 = _z_func_full["2010-01-01":]
    n_nan_2010 = int(z_2010.isna().sum())
    ck("W11c Contrarian Z-Score kein NaN ab 2010",
       n_nan_2010 == 0,
       f"{n_nan_2010} NaN-Werte ab 2010")

    # d) Nullstd → NaN, kein Division-Error
    # std.replace(0, np.nan) stellt sicher dass keine Inf/Null-Division vorkommt
    n_inf = int(np.isinf(_z_func_full.replace(np.nan, 0.0)).sum())
    ck("W11d Contrarian Z-Score: keine Inf-Werte (Nullstd erzeugt NaN, nicht Inf)",
       n_inf == 0,
       f"{n_inf} Inf-Werte in der gesamten Reihe")

    # ── W12 – Benchmark: Drift-bedingtes Rebalancing ──────────────────────────
    bm_turnover = float(ports["Benchmark"]["turnover"].sum())
    bm_rebal_days = ports["Benchmark"][ports["Benchmark"]["turnover"] > 1e-9]
    n_bm_rebal = len(bm_rebal_days)

    # a) Gesamt-Turnover > 0
    ck("W12a Benchmark hat Drift-bedingtes Rebalancing (Turnover > 0)",
       bm_turnover > 0,
       f"Gesamt-Turnover: {bm_turnover:.4f}")

    # b) Mehr als 10 Rebalancing-Tage
    ck("W12b Benchmark: mehr als 10 Rebalancing-Tage gesamt",
       n_bm_rebal > 10,
       f"{n_bm_rebal} Rebalancing-Tage")

    # c) Rebalancing in mindestens 3 verschiedenen Jahren
    bm_rebal_years = set(bm_rebal_days.index.year)
    ck("W12c Benchmark: Rebalancing in ≥ 3 verschiedenen Jahren",
       len(bm_rebal_years) >= 3,
       f"Jahre mit Rebalancing: {sorted(bm_rebal_years)[:5]}{'...' if len(bm_rebal_years) > 5 else ''}")

    # d) Mit Band > 0: Benchmark hat weniger Turnover als Band=0-Version
    # w_bm wird aus den gespeicherten Tagesgewichten im port rekonstruiert.
    if band_bm > 0:
        bm_w1_chk = float(ports["Benchmark"]["a1_weight"].iloc[0])
        bm_w2_chk = float(ports["Benchmark"]["a2_weight"].iloc[0])
        _w_bm_chk = pd.DataFrame(
            {ticker1: bm_w1_chk, ticker2: bm_w2_chk},
            index=m_prices.index,
        )
        p_bm_0band  = _backtest(daily_close, daily_open, _w_bm_chk, tc_bps, band=0.0)
        to_bm_0band = float(p_bm_0band["turnover"].sum())
        ck("W12d Benchmark: Band > 0 reduziert Turnover vs. Band=0",
           bm_turnover <= to_bm_0band,
           f"Mit Band {band_bm*100:.0f}%: {bm_turnover:.4f}  |  Band=0: {to_bm_0band:.4f}")
    else:
        ck("W12d Benchmark Band-Vergleich übersprungen (Band=0%)", True, "Band=0 → Test entfällt")

    # ── W13 – T+1-Ausführungssplit ────────────────────────────────────────────
    _close_chk = daily_close["2005-01-01":].copy()
    _open_chk  = daily_open.reindex(_close_chk.index)
    _prev_c    = _close_chk.shift(1)
    _r_ovn     = (_open_chk - _prev_c) / _prev_c
    _r_intr    = (_close_chk - _open_chk) / _open_chk
    _r_cc_chk  = _close_chk.pct_change()
    _p_mom_chk = ports["Momentum"]
    _rebal_days_w13 = _p_mom_chk[_p_mom_chk["turnover"] > 1e-9].index[:10]

    _split_ok_n = 0
    _split_tot  = 0
    for _d in _rebal_days_w13:
        if _d not in _close_chk.index or _d == _close_chk.index[0]:
            continue
        _i = _close_chk.index.get_loc(_d)
        _pv_before  = _p_mom_chk["portfolio_value"].iloc[_i - 1]
        _pv_after   = _p_mom_chk["portfolio_value"].iloc[_i]
        _w_prev     = np.array([_p_mom_chk["a1_weight"].iloc[_i - 1],
                                 _p_mom_chk["a2_weight"].iloc[_i - 1]])
        _d_prev     = _close_chk.index[_i - 1]
        _prev_rebal = _p_mom_chk["turnover"].iloc[_i - 1] > 1e-9
        _r_prev     = (_r_intr.loc[_d_prev].fillna(0.0).values if _prev_rebal
                       else _r_cc_chk.loc[_d_prev].fillna(0.0).values)
        _w_eod      = _w_prev * (1.0 + _r_prev)
        _s          = _w_eod.sum()
        _w_drift    = _w_eod / _s if _s > 1e-9 else _w_prev.copy()
        _w_new      = np.array([_p_mom_chk["a1_weight"].iloc[_i],
                                 _p_mom_chk["a2_weight"].iloc[_i]])
        _ovn_v      = _r_ovn.loc[_d].fillna(0.0).values
        _intra_v    = _r_intr.loc[_d].fillna(0.0).values
        _tc_d       = float(_p_mom_chk["tc_cost"].iloc[_i])
        _ovn_r      = float(np.dot(_w_drift, _ovn_v))
        _intra_r    = float(np.dot(_w_new,   _intra_v))
        _pv_exp     = _pv_before * (1.0 + _ovn_r) * (1.0 - _tc_d) * (1.0 + _intra_r)
        _split_tot += 1
        if abs(_pv_after - _pv_exp) < 1e-9:
            _split_ok_n += 1

    ck("W13a T+1-Split: Rebalancing-Tage — Overnight mit alten, Intraday mit neuen Gewichten",
       _split_tot > 0 and _split_ok_n == _split_tot,
       f"{_split_ok_n}/{_split_tot} Rebalancing-Tage rechnerisch korrekt")

    # b) Normale (nicht-Rebalancing) Tage: PV = PV_prev × (1 + dot(w_drift, r_cc))
    _normal_days = _p_mom_chk[_p_mom_chk["turnover"] < 1e-9].index[1:6]
    _norm_ok_n   = 0
    _norm_tot    = 0
    for _d in _normal_days:
        if _d not in _close_chk.index or _d == _close_chk.index[0]:
            continue
        _i = _close_chk.index.get_loc(_d)
        _pv_before  = _p_mom_chk["portfolio_value"].iloc[_i - 1]
        _pv_after   = _p_mom_chk["portfolio_value"].iloc[_i]
        # Normale Tage: w_trade = current_w (keine Änderung), TC=0, ret = dot(w_drift, r_cc)
        _w_prev     = np.array([_p_mom_chk["a1_weight"].iloc[_i - 1],
                                 _p_mom_chk["a2_weight"].iloc[_i - 1]])
        _d_prev     = _close_chk.index[_i - 1]
        _prev_rebal_n = _p_mom_chk["turnover"].iloc[_i - 1] > 1e-9
        _r_prev_n   = (_r_intr.loc[_d_prev].fillna(0.0).values if _prev_rebal_n
                       else _r_cc_chk.loc[_d_prev].fillna(0.0).values)
        _w_eod_n    = _w_prev * (1.0 + _r_prev_n)
        _sn         = _w_eod_n.sum()
        _w_drift_n  = _w_eod_n / _sn if _sn > 1e-9 else _w_prev.copy()
        _r_cc_d     = _r_cc_chk.loc[_d].fillna(0.0).values
        _r_exp_n    = float(np.dot(_w_drift_n, _r_cc_d))
        # cost = 0 auf normalen Tagen (Turnover=0 → TC=0)
        _tc_n       = float(_p_mom_chk["tc_cost"].iloc[_i])
        _pv_exp_n   = _pv_before * (1.0 + _r_exp_n) * (1.0 - _tc_n)
        _norm_tot  += 1
        if abs(_pv_after - _pv_exp_n) < 1e-9:
            _norm_ok_n += 1

    ck("W13b T+1-Split: Normale Tage — Close-to-Close mit gedrifteten Gewichten",
       _norm_tot == 0 or _norm_ok_n == _norm_tot,
       f"{_norm_ok_n}/{_norm_tot} normale Tage rechnerisch korrekt")

    # =========================================================================
    # GRUPPE D — FILTERLOGIK  (W14, W17–W19, W21)
    # =========================================================================

    _fs_trade = filter_s["2005-01-01":]
    _z_idx    = z_mon_s.index
    _ms_idx   = mom_sig.index

    # ── W14 – Filter-Hierarchie vollständig ──────────────────────────────────
    # a) F1 dominiert wenn Z < crash_thr UND mom > 0
    _f1_conflict_ok = True
    _f1_conflict_n  = 0
    _f1_conflict_ex = []
    # b) F2 dominiert wenn Z >= crash_thr UND mom <= 0 (F2 beats F3)
    _f2_conflict_ok = True
    _f2_conflict_n  = 0
    _f2_conflict_ex = []
    # c) F3 nur wenn Z >= crash_thr UND mom > 0
    _f3_conflict_ok = True
    _f3_conflict_n  = 0
    _f3_conflict_ex = []

    for _d in _fs_trade.index:
        if _d not in _z_idx or _d not in _ms_idx:
            continue
        _zv = float(z_mon_s.loc[_d]) if _d in z_mon_s.index else float("nan")
        _mv = float(mom_sig.loc[_d])  if _d in mom_sig.index else float("nan")
        if np.isnan(_zv) or np.isnan(_mv):
            continue
        _fv = _fs_trade.loc[_d]

        # F1: Z < crash_thr AND mom > 0 → muss F1_crash sein
        if _zv < crash_thr and _mv > 0:
            _f1_conflict_n += 1
            if _fv != "F1_crash":
                _f1_conflict_ok = False
                _f1_conflict_ex.append(f"{_d.date()}: Z={_zv:.2f} m={_mv:.2%} → {_fv}")

        # F2: Z >= crash_thr AND mom <= 0 → muss F2_momentum sein (nicht F3)
        if _zv >= crash_thr and _mv <= 0:
            _f2_conflict_n += 1
            if _fv != "F2_momentum":
                _f2_conflict_ok = False
                _f2_conflict_ex.append(f"{_d.date()}: Z={_zv:.2f} m={_mv:.2%} → {_fv}")

        # F3: F3_volscale nur wenn Z >= crash_thr AND mom > 0
        if _fv == "F3_volscale":
            _f3_conflict_n += 1
            if _zv < crash_thr or _mv <= 0:
                _f3_conflict_ok = False
                _f3_conflict_ex.append(f"{_d.date()}: Z={_zv:.2f} m={_mv:.2%}")

    ck("W14a Filter-Hierarchie: F1 dominiert über positives Momentum",
       _f1_conflict_ok,
       (f"Alle {_f1_conflict_n} F1-Konflikte korrekt" if _f1_conflict_ok and _f1_conflict_n > 0
        else f"Keine Konflikte" if _f1_conflict_n == 0
        else f"Verletzt: {'; '.join(_f1_conflict_ex[:3])}"))
    ck("W14b Filter-Hierarchie: F2 dominiert über F3 (Z≥thr, mom≤0)",
       _f2_conflict_ok,
       (f"Alle {_f2_conflict_n} F2-Fälle korrekt" if _f2_conflict_ok
        else f"Verletzt: {'; '.join(_f2_conflict_ex[:3])}"))
    ck("W14c Filter-Hierarchie: F3 nur bei Z≥thr UND mom>0",
       _f3_conflict_ok,
       (f"Alle {_f3_conflict_n} F3-Monate korrekt" if _f3_conflict_ok
        else f"Verletzt: {'; '.join(_f3_conflict_ex[:3])}"))

    # ── W17 – Filter-1/2-Gewichte exakt 0.20/0.80 ────────────────────────────
    _f12_months = _fs_trade[_fs_trade.isin(["F1_crash", "F2_momentum"])]
    _w_wrong_n  = 0
    _w_wrong_ex = []
    for _d in _f12_months.index:
        if _d not in w_mom.index:
            continue
        _w1 = float(w_mom.loc[_d, ticker1])
        _w2 = float(w_mom.loc[_d, ticker2])
        if abs(_w1 - 0.20) > 1e-9 or abs(_w2 - 0.80) > 1e-9:
            _w_wrong_n += 1
            _w_wrong_ex.append(f"{_d.date()}: {_w1:.4f}/{_w2:.4f}")
    ck(f"W17a F1/F2-Gewichte: {ticker1}=0.20, {ticker2}=0.80 in allen Crash-/Neg.-Momentum-Monaten",
       _w_wrong_n == 0,
       (f"Alle {len(_f12_months)} F1/F2-Monate korrekt" if _w_wrong_n == 0
        else f"{_w_wrong_n} falsche: {'; '.join(_w_wrong_ex[:3])}"))

    # b) Einige F3-Monate haben Gewichte ≠ 0.20 (Vol-Scaling ist aktiv)
    if len(f3_dates) > 0:
        _f3_w1 = w_mom.reindex(f3_dates)[ticker1]
        _f3_non_def = int((_f3_w1 - 0.20).abs().gt(1e-6).sum())
        _pct_non_def = _f3_non_def / len(f3_dates) * 100
        ck("W17b F3-Gewichte: mindestens 30% der Monate haben vol-skalierte Gewichte",
           _pct_non_def >= 30.0,
           f"{_f3_non_def}/{len(f3_dates)} Monate ({_pct_non_def:.1f}%) mit Gewicht ≠ 0.20")

        # c) Alle F3-Gewichte in [0.0, 1.0]
        _f3_min = float(_f3_w1.min())
        _f3_max = float(_f3_w1.max())
        ck("W17c F3-Gewichte strikt in [0.0, 1.0]",
           _f3_min >= 0.0 and _f3_max <= 1.0,
           f"Min: {_f3_min:.4f}  |  Max: {_f3_max:.4f}")

    # ── W18 – Contrarian Richtungskonsistenz ──────────────────────────────────
    _z_con_trade = z_con_d["2005-01-01":]
    _w_con_trade = w_con["2005-01-01":]
    _dir_fail = 0
    _dir_total = 0
    _z_vals_for_spearman = []
    _w_vals_for_spearman = []

    for _cd in _w_con_trade.index:
        _z_at = _z_con_trade[_z_con_trade.index <= _cd]
        if len(_z_at) == 0 or pd.isna(_z_at.iloc[-1]):
            continue
        _z_val = float(_z_at.iloc[-1])
        _w_val = float(_w_con_trade.loc[_cd, ticker1])
        _dir_total += 1
        _z_vals_for_spearman.append(_z_val)
        _w_vals_for_spearman.append(_w_val)
        if _z_val > 1e-6 and _w_val >= w_neutral + 1e-6:
            _dir_fail += 1
        elif _z_val < -1e-6 and _w_val <= w_neutral - 1e-6:
            _dir_fail += 1

    # a) Richtungslogik verletzt?
    ck("W18a Contrarian Richtungskonsistenz: Z>0→w<w_neutral, Z<0→w>w_neutral",
       _dir_fail == 0,
       (f"Alle {_dir_total} Monate konsistent" if _dir_fail == 0
        else f"{_dir_fail}/{_dir_total} verletzen Richtungslogik"))

    # b) Monotonizität via Spearman-Korrelation Z vs. w (sollte stark negativ sein)
    if len(_z_vals_for_spearman) >= 10:
        _z_rank = pd.Series(_z_vals_for_spearman).rank()
        _w_rank = pd.Series(_w_vals_for_spearman).rank()
        _n_sp   = len(_z_rank)
        _sp_cor = float(1 - 6 * ((_z_rank - _w_rank)**2).sum() / (_n_sp * (_n_sp**2 - 1)))
        ck("W18b Monotonizität: Spearman(Z, w) < −0.5 (negativ korreliert)",
           _sp_cor < -0.5,
           f"Spearman r = {_sp_cor:.3f}  (erwartet < −0.5)")
    else:
        ck("W18b Spearman-Korrelation übersprungen", True, "Zu wenig Monate für Test")

    # c) Clipping selten (<5% der Monate)
    _clip_count = 0
    for _z_v, _w_v in zip(_z_vals_for_spearman, _w_vals_for_spearman):
        _w_raw = w_neutral + alpha * np.tanh(-beta * _z_v)
        if abs(_w_raw - _w_v) > 1e-6:
            _clip_count += 1
    _pct_clipped = _clip_count / max(1, _dir_total) * 100
    # Erwartete Clipping-Rate ist PARAMETERABHÄNGIG:
    # Bei w_neutral=0.80, alpha=0.50, beta=0.75 tritt Clipping auf wenn Z < −0.57
    # (tanh(−β·Z) > (1−w_neutral)/alpha = 0.4). Bei standardnormaler Z-Verteilung
    # sind ~29% der Werte darunter → "< 5%" wäre ein falsches Kriterium.
    # Sinnvoller Test: < 75% geclipt (sonst dominiert alpha die Formel vollständig).
    _expected_clip_pct = float(np.mean([
        abs((w_neutral + alpha * np.tanh(-beta * _z)) -
            np.clip(w_neutral + alpha * np.tanh(-beta * _z), 0.0, 1.0)) > 1e-6
        for _z in np.linspace(-3.0, 3.0, 1000)
    ])) * 100
    ck("W18c Clipping-Rate < 75% (tanh-Formel nicht durch Clip dominiert)",
       _pct_clipped < 75.0,
       f"{_clip_count}/{_dir_total} Monate geclipt ({_pct_clipped:.1f}%)  "
       f"[theoret. bei Normalvert.: ≈{_expected_clip_pct:.0f}%]",
       warn=True)

    # ── W19 – Filterlogik vollständig ─────────────────────────────────────────
    _known     = {"F1_crash", "F2_momentum", "F3_volscale"}
    _unknown_n = int((~_fs_trade.isin(_known)).sum())
    _f1n = int((_fs_trade == "F1_crash").sum())
    _f2n = int((_fs_trade == "F2_momentum").sum())
    _f3n = int((_fs_trade == "F3_volscale").sum())
    _total_trade = len(_fs_trade)

    # a) F1+F2+F3 = Gesamt, kein Warmup
    ck("W19a Filterlogik vollständig: F1+F2+F3 = alle Monate 2005+",
       _unknown_n == 0 and (_f1n + _f2n + _f3n == _total_trade),
       (f"F1={_f1n}  F2={_f2n}  F3={_f3n}  Σ={_f1n+_f2n+_f3n}/{_total_trade}" if _unknown_n == 0
        else f"{_unknown_n} unbekannte Einträge gefunden"))

    # b) F3 ist häufigster Regimetyp (>40% in 20-Jahr-Bullmarkt)
    _f3_pct = _f3n / _total_trade * 100
    ck("W19b F3 ist häufigster Filter-Typ (> 40% in 20-J-Bullmarkt)",
       _f3_pct > 40.0,
       f"F1={_f1n/_total_trade*100:.0f}% ({_f1n})  F2={_f2n/_total_trade*100:.0f}% ({_f2n})  "
       f"F3={_f3_pct:.0f}% ({_f3n})")

    # c) F1 in Krisenjahren, F1 oder F2 in 2020
    _f_by_year = _fs_trade.groupby(_fs_trade.index.year)
    _f1_years  = set()
    _f12_2020  = 0
    for _yr, _grp in _f_by_year:
        if (_grp == "F1_crash").any():
            _f1_years.add(_yr)
        if _yr == 2020 and (_grp.isin(["F1_crash", "F2_momentum"])).any():
            _f12_2020 = int((_grp.isin(["F1_crash", "F2_momentum"])).sum())
    # F1 (Crash-Filter) reagiert nur auf PLÖTZLICHE Extremcrashs (Z << crash_thr),
    # gemessen am letzten Handelstag des Monats. Ob F1 in einem bestimmten Krisenjahr
    # feuert, hängt stark von crash_thr und zscore_win ab: Bei graduellen Bärenmärkten
    # (hohe Vorvolatilität → großes σ → Z moderat) kann F1 ausbleiben, obwohl der Markt
    # insgesamt stark fällt. Das ist kein Fehler, sondern Design.
    # Sinnvoller Korrektheitsheck: F1 ist mindestens einmal aktiv (Filter nicht tot)
    # und feuert ausschließlich bei echten Z-Unterschreitungen (W21 prüft Konsistenz).
    _crisis_years = {2008, 2009, 2020, 2022}
    _f1_in_any_crisis = bool(_f1_years & _crisis_years)
    ck("W19c F1 mindestens einmal aktiv (Crash-Filter nicht tot)",
       len(_f1_years) > 0,
       f"F1-aktive Jahre: {sorted(_f1_years)}"
       + (f"  |  davon Krisenperioden: {sorted(_f1_years & _crisis_years)}"
          if _f1_in_any_crisis else "  |  (keine Überschneidung mit 2008/09/20/22 — parametrierungsabhängig)"),
       warn=False)
    ck("W19d F1/F2 im COVID-Jahr 2020 vorhanden",
       _f12_2020 > 0,
       f"{_f12_2020} Crash-/Momentum-Monate in 2020", warn=True)

    # ── W21 – Filter-Signal-Konsistenz ────────────────────────────────────────
    _sig_incon = 0
    _sig_ex    = []
    for _d in _fs_trade.index:
        _fval = _fs_trade.loc[_d]
        _zval = float(z_mon_s.loc[_d]) if _d in z_mon_s.index else float("nan")
        _mval = float(mom_sig.loc[_d])  if _d in mom_sig.index else float("nan")
        if np.isnan(_zval) or np.isnan(_mval):
            continue
        _ok = True
        if _fval == "F1_crash"     and not (_zval < crash_thr):
            _ok = False
        elif _fval == "F2_momentum" and not (_mval <= 0 and _zval >= crash_thr):
            _ok = False
        elif _fval == "F3_volscale" and not (_mval > 0 and _zval >= crash_thr):
            _ok = False
        if not _ok:
            _sig_incon += 1
            _sig_ex.append(f"{_d.date()} {_fval}: Z={_zval:.2f} mom={_mval:.2%}")

    ck("W21a Filter-Signal-Konsistenz: alle Labels stimmen mit Z/Momentum überein",
       _sig_incon == 0,
       (f"Alle {len(_fs_trade)} Monate konsistent" if _sig_incon == 0
        else f"{_sig_incon} inkonsistent: {'; '.join(_sig_ex[:3])}"))

    # b) Informativer Regimen-Check für bekannte Marktereignisse.
    #    Der Crash-Filter F1 löst NUR bei plötzlichen Extremcrashs aus (Z << crash_thr).
    #    2008 war ein gradueller Bärenmarkt: erhöhte Vorvolatilität (σ_prev) dämpft Z.
    #    F2_momentum ist dort korrekt: Trend war negativ, aber kein Sigma-Extremereignis.
    #    Wir zeigen den tatsächlichen Filter — kein hartes Fail-Kriterium hier.
    _event_checks = [
        ("2008-10-31", "Lehman-Nachwirkung (grad. Bärenmarkt → oft F2)"),
        ("2020-03-31", "COVID-Crash (plötzl. Extremcrash → oft F1)"),
        ("2022-09-30", "Zinsschock 2022 (Bärenmarkt → oft F2)"),
        ("2009-02-27", "GFC-Talsohle (Erholungsphase → F2/F3)"),
    ]
    _event_details = []
    for _ev_str, _ev_label in _event_checks:
        _ev_ts  = pd.Timestamp(_ev_str)
        _ev_idx = _fs_trade.index[_fs_trade.index <= _ev_ts]
        if len(_ev_idx) > 0:
            _ev_d = _ev_idx[-1]
            _event_details.append(f"{_ev_d.date()} ({_ev_label}): {_fs_trade.loc[_ev_d]}")
    ck("W21b Regime-Verteilung an bekannten Marktereignissen (informativ)",
       True,
       " | ".join(_event_details) if _event_details else "Keine Daten")

    # c) 2013-12-31 sollte F3_volscale sein (QE-Bullmarkt)
    _tgt_qe = pd.Timestamp("2013-12-31")
    _closest_qe = _fs_trade.index[_fs_trade.index <= _tgt_qe]
    if len(_closest_qe) > 0:
        _cqe = _closest_qe[-1]
        ck("W21c ~2013-12 ist F3_volscale (QE-Bullmarkt)",
           _fs_trade.loc[_cqe] == "F3_volscale",
           f"{_cqe.date()}: {_fs_trade.loc[_cqe]}", warn=True)

    # d) Anzahl Filter-Zustandswechsel plausibel (> 10, < 200)
    _transitions = int((_fs_trade != _fs_trade.shift(1)).sum()) - 1  # -1 für ersten Eintrag
    ck("W21d Anzahl Filter-Zustandswechsel plausibel (>10, <200)",
       10 < _transitions < 200,
       f"{_transitions} Zustandswechsel über ~20 Jahre")

    # =========================================================================
    # GRUPPE E — CONTRARIAN-STRATEGIE  (W15, erweitertes W18)
    # =========================================================================

    # ── W15 – tanh-Formel Eigenschaften ──────────────────────────────────────
    # a) Informative: tanh-Rohwerte in Panikphasen
    _panic_dates = ["2020-03-31", "2008-10-31", "2009-02-27", "2022-09-30"]
    _panic_detail = []
    for _pd_str in _panic_dates:
        try:
            _pd_ts  = pd.Timestamp(_pd_str)
            _closest_p = z_con_d.index[z_con_d.index <= _pd_ts]
            if len(_closest_p) == 0:
                continue
            _pd_act = _closest_p[-1]
            _z_raw  = float(z_con_d.loc[_pd_act])
            if pd.isna(_z_raw):
                continue
            _w_raw  = w_neutral + alpha * np.tanh(-beta * _z_raw)
            _w_clip = float(np.clip(_w_raw, 0.0, 1.0))
            _tag    = " ← GECLIPT" if abs(_w_raw - _w_clip) > 0.001 else ""
            _panic_detail.append(f"{_pd_act.date()} Z={_z_raw:.2f} w_raw={_w_raw:.3f}{_tag}")
        except Exception:
            pass
    ck("W15a tanh-Rohwerte in Extremphasen (informativ)",
       True,
       " | ".join(_panic_detail) if _panic_detail else "Keine Extremdaten")

    # b) Z=0 → w = w_neutral exakt
    _w_at_z0 = w_neutral + alpha * np.tanh(0.0)
    ck("W15b tanh(0)=0 → w(Z=0) = w_neutral exakt",
       abs(_w_at_z0 - w_neutral) < 1e-12,
       f"w_neutral={w_neutral}  w(Z=0)={_w_at_z0:.15f}")

    # c) |tanh(-β·Z)| ≤ 1 → Gewicht begrenzt in [w_neutral-α, w_neutral+α] vor Clip
    _z_con_valid = z_con_d["2005-01-01":].dropna()
    if len(_z_con_valid) > 0:
        _w_unclipped = w_neutral + alpha * np.tanh(-beta * _z_con_valid)
        _max_exceedance = float((_w_unclipped - w_neutral).abs().max())
        ck("W15c Gewicht (vor Clip) in [w_neutral-α, w_neutral+α]",
           _max_exceedance <= alpha + 1e-9,
           f"Max |w_raw - w_neutral| = {_max_exceedance:.6f}  ≤ α={alpha}")

    # d) Bekannter Z-Wert: Prüfe ob Gewicht zur tanh-Formel passt
    # Suche ein Datum mit Z ≈ 2.0
    _z_test_cand = _z_con_valid[(_z_con_valid - 2.0).abs() < 0.3]
    if len(_z_test_cand) > 0:
        _z_test_date = _z_test_cand.index[0]
        # Monatsebene: nächster Monatsendes in w_con suchen
        _w_con_2005  = w_con["2005-01-01":]
        _m_cand      = _w_con_2005.index[_w_con_2005.index >= _z_test_date]
        if len(_m_cand) > 0:
            _m_date     = _m_cand[0]
            _z_at_m     = float(z_con_d.asof(_m_date)) if hasattr(z_con_d, "asof") else float("nan")
            if not np.isnan(_z_at_m):
                _w_exp_d = float(np.clip(w_neutral + alpha * np.tanh(-beta * _z_at_m), 0.0, 1.0))
                _w_act_d = float(_w_con_2005.loc[_m_date, ticker1])
                ck("W15d Contrarian-Gewicht stimmt mit tanh-Formel bei bekanntem Z",
                   abs(_w_exp_d - _w_act_d) < 1e-6,
                   f"{_m_date.date()} Z={_z_at_m:.3f}  w_exp={_w_exp_d:.6f}  w_akt={_w_act_d:.6f}")
            else:
                ck("W15d tanh-Formelprüfung bei Z≈2", True, "Kein gültiger Z-Wert am Monatsdatum")
        else:
            ck("W15d tanh-Formelprüfung bei Z≈2", True, "Kein Monatsende nach Testdatum gefunden")
    else:
        ck("W15d tanh-Formelprüfung bei Z≈2", True, "Kein Datum mit Z≈2.0 im Datensatz")

    # =========================================================================
    # GRUPPE F — QUALITÄTS-CHECKS  (W16, W20, W22)
    # =========================================================================

    # ── W16 – Datenlücken und Datenqualität ───────────────────────────────────
    _r_abs  = daily_close["2005-01-01":].pct_change().abs()
    _max_gap = float(max(_r_abs[ticker1].max(), _r_abs[ticker2].max()))

    # a) Keine Datenlücke > 20%
    _gap_ex = []
    for _t in [ticker1, ticker2]:
        for _gd, _gv in _r_abs[_t].nlargest(3).items():
            if _gv > 0.05:
                _gap_ex.append(f"{_t} {_gd.date()}: {_gv*100:.1f}%")
    ck(f"W16a Keine Datenlücke > 20% ({ticker1} & {ticker2})",
       _max_gap < 0.20,
       (f"Max: {_max_gap*100:.2f}%  |  Größte: {', '.join(_gap_ex[:4])}"
        if _gap_ex else f"Max: {_max_gap*100:.2f}%"))

    # b) Kein Handelstag mit Preis ≤ 0
    _min_price = float(daily_close["2005-01-01":].min().min())
    ck("W16b Alle Preise positiv (kein Handelstag mit Preis ≤ 0)",
       _min_price > 0.0,
       f"Min Preis: {_min_price:.4f}")

    # c) Handelstage pro Jahr in [245, 260]
    _dc_2005 = daily_close["2005-01-01":"2024-12-31"]
    _days_per_year = _dc_2005.groupby(_dc_2005.index.year).size()
    _min_dpy = int(_days_per_year.min())
    _max_dpy = int(_days_per_year.max())
    _bad_years = _days_per_year[(_days_per_year < 245) | (_days_per_year > 260)]
    ck("W16c Handelstage pro Jahr in [245, 260] (2005–2024)",
       len(_bad_years) == 0,
       f"Min: {_min_dpy}  Max: {_max_dpy}"
       + (f"  |  Ausreißer: {dict(_bad_years)}" if len(_bad_years) > 0 else "  — alle Jahre OK"))

    # d) Beide Assets haben gleich viele Handelstage (dropna ist konsistent)
    _len_t1 = int(daily_close[ticker1]["2005-01-01":].notna().sum())
    _len_t2 = int(daily_close[ticker2]["2005-01-01":].notna().sum())
    ck("W16d Beide Assets haben gleich viele Handelstage",
       _len_t1 == _len_t2,
       f"{ticker1}: {_len_t1}  |  {ticker2}: {_len_t2}")

    # ── W20 – Rebalancing-Häufigkeit plausibel ────────────────────────────────
    _years_range   = list(range(2005, 2026))
    _rebal_summary = {}
    for _sn, _pt in ports.items():
        _rpy = [int((_pt[_pt.index.year == _yr]["turnover"] > 1e-9).sum())
                for _yr in _years_range]
        _rebal_summary[_sn] = _rpy

    _rebal_ok = True
    _rebal_details = []
    for _sn, _cnts in _rebal_summary.items():
        _max_c = max(_cnts) if _cnts else 0
        _min_c = min(_cnts) if _cnts else 0
        _avg_c = float(np.mean(_cnts)) if _cnts else 0.0
        if _max_c > 260:
            _rebal_ok = False
        _rebal_details.append(f"{_sn}: ⌀{_avg_c:.1f}/J (min {_min_c}, max {_max_c})")

    # a) Max ≤ 260 Rebalancings pro Jahr
    ck("W20a Rebalancing-Häufigkeit ≤ 260/Jahr (jede Strategie)",
       _rebal_ok,
       " | ".join(_rebal_details))

    # b) Mit Band > 0: ≤ 52 Rebalancings/Jahr im Durchschnitt (max 1×/Woche)
    if band_pct > 0:
        _all_avg = [float(np.mean(v)) for v in _rebal_summary.values()]
        _max_avg = max(_all_avg)
        ck("W20b Ø Rebalancings/Jahr ≤ 52 bei Band > 0 (max 1×/Woche)",
           _max_avg <= 52,
           f"Max Ø über alle Strategien: {_max_avg:.1f}/Jahr")

    # c) Monotonizität: doppeltes Band → weniger Rebalancings (für Momentum)
    if band_pct >= 0.05:
        _dbl_band = min(band_pct * 2, 0.40)
        _p_dbl    = _backtest(daily_close, daily_open, w_mom, tc_bps, band=_dbl_band)
        _to_dbl   = float((_p_dbl["turnover"] > 1e-9).sum())
        _to_cur   = float((ports["Momentum"]["turnover"] > 1e-9).sum())
        ck(f"W20c Doppeltes Band ({_dbl_band*100:.0f}%) hat weniger Rebalancings als Band={band_pct*100:.0f}%",
           _to_dbl <= _to_cur,
           f"Band={band_pct*100:.0f}%: {_to_cur:.0f} Tage  |  Band={_dbl_band*100:.0f}%: {_to_dbl:.0f} Tage")

    # ── W22 – Strategie-Qualität ──────────────────────────────────────────────
    _cagr_bm  = m_bm["CAGR"]
    _cagr_mom = m_mom["CAGR"]
    _cagr_con = m_con["CAGR"]
    _alpha_mom = _cagr_mom - _cagr_bm
    _alpha_con = _cagr_con - _cagr_bm
    _alpha_ok  = _alpha_mom >= 0 or _alpha_con >= 0

    # Brutto-Alpha (vor TC) über Kosten-Drag rekonstruieren:
    # CAGR_brutto = CAGR_netto + Kosten-Drag  →  α_brutto = CAGR_brutto_strat − CAGR_brutto_bm
    _gcagr_bm  = _cagr_bm  + m_bm["Kosten-Drag"]
    _gcagr_mom = _cagr_mom + m_mom["Kosten-Drag"]
    _gcagr_con = _cagr_con + m_con["Kosten-Drag"]
    _galpha_mom = _gcagr_mom - _gcagr_bm
    _galpha_con = _gcagr_con - _gcagr_bm
    _gross_ok   = _galpha_mom >= 0 or _galpha_con >= 0

    # a) Mind. eine Strategie hat positives Alpha
    _detail_22a = (f"Netto: Mom α={_alpha_mom*100:+.2f}%  Con α={_alpha_con*100:+.2f}%  |  "
                   f"Brutto: Mom α={_galpha_mom*100:+.2f}%  Con α={_galpha_con*100:+.2f}%")
    if not _alpha_ok and _gross_ok:
        _detail_22a += "  → TC-Drag verursacht Netto-Unterperformance"
    ck("W22a Mindestens eine Strategie übertrifft Benchmark (α ≥ 0)",
       _alpha_ok,
       _detail_22a,
       warn=True)

    # b) Mind. eine Strategie hat Sharpe ≥ Benchmark-Sharpe
    _sh_bm  = m_bm["Sharpe"]
    _sh_mom = m_mom["Sharpe"]
    _sh_con = m_con["Sharpe"]
    _sharpe_ok = (_sh_mom >= _sh_bm or _sh_con >= _sh_bm)
    # Brutto-Sharpe approximieren: Sharpe_brutto ≈ Sharpe_netto + Kosten-Drag / Vol
    _gsh_bm  = _sh_bm  + m_bm["Kosten-Drag"]  / m_bm["Vol"]  if m_bm["Vol"]  > 0 else _sh_bm
    _gsh_mom = _sh_mom + m_mom["Kosten-Drag"] / m_mom["Vol"] if m_mom["Vol"] > 0 else _sh_mom
    _gsh_con = _sh_con + m_con["Kosten-Drag"] / m_con["Vol"] if m_con["Vol"] > 0 else _sh_con
    _gsharpe_ok = (_gsh_mom >= _gsh_bm or _gsh_con >= _gsh_bm)
    _detail_22b = (f"Netto: BM={_sh_bm:.3f}  Mom={_sh_mom:.3f}  Con={_sh_con:.3f}  |  "
                   f"Brutto: BM={_gsh_bm:.3f}  Mom={_gsh_mom:.3f}  Con={_gsh_con:.3f}")
    if not _sharpe_ok and _gsharpe_ok:
        _detail_22b += "  → TC-Drag verursacht Netto-Unterperformance"
    ck("W22b Mindestens eine Strategie hat Sharpe ≥ Benchmark-Sharpe",
       _sharpe_ok,
       _detail_22b,
       warn=True)

    # c) Beide Strategien haben Max Drawdown < 100% (kein Totalverlust)
    _mdd_mom = float(m_mom["Max DD"])
    _mdd_con = float(m_con["Max DD"])
    ck("W22c Beide Strategien: Max Drawdown < 100% (kein Totalverlust)",
       _mdd_mom > -1.0 and _mdd_con > -1.0,
       f"Mom DD={_mdd_mom*100:.1f}%  Con DD={_mdd_con*100:.1f}%")

    return checks


# =============================================================================
# §10  LIVE-DASHBOARD  main()  ✦ APP-INFRASTRUKTUR — weniger kritisch
# =============================================================================
# main() orchestriert die gesamte Streamlit-App. Die Reihenfolge ist:
#
#   §10a  Sidebar          → User stellt Parameter ein (Slider, Textfelder)
#   §10b  Datenberechnung  → Strategiegewichte + Backtests werden berechnet
#   §10c  KPI-Tabelle      → CAGR, Sharpe etc. farbkodiert angezeigt
#   §10d  Charts & Tabelle → Alle Plotly-Charts + Monatstabelle
#   §10e  Analyse-Export   → Markdown-Bericht für KI-Analyse generieren
#   §10f  Logik-Check      → W1–W22 mathematische Selbstprüfungen
#
# Streamlit führt main() bei JEDER User-Interaktion von oben nach unten aus.
# @st.cache_data-Funktionen werden nur bei Parameteränderung neu berechnet.
# =============================================================================

def main():

    # ── §10a SIDEBAR: PARAMETER ───────────────────────────────────────────────
    # Streamlit rendert die Sidebar zuerst, damit alle Parameter-Variablen
    # (ticker1, tc_bps, band_bm/mom/con usw.) für den Rest der Funktion verfügbar sind.
    with st.sidebar:
        st.header("Parameter")
        st.markdown("Alle Änderungen aktualisieren die Charts **sofort**.")

        with st.expander("Assets", expanded=False):
            st.caption("Standard: SPY (S&P 500) & GLD (Gold)")
            ticker1 = st.text_input("Asset 1 (Aktien / Equity)", value="SPY").upper().strip()
            ticker2 = st.text_input("Asset 2 (Hedge / Gold)", value="GLD").upper().strip()
            bm_w1   = st.slider("Benchmark-Gewicht Asset 1 (%)", 50, 95, 80, 5,
                                 help="Statisches Zielgewicht des Benchmarks") / 100
            if ticker1 == ticker2:
                st.error("Asset 1 und Asset 2 müssen unterschiedlich sein!")
                st.stop()

        with st.expander("Allgemein", expanded=True):
            tc_bps   = st.slider("Transaktionskosten (bps/Seite)", 0, 50, 5, 1)
            band_bm  = st.slider("Rebalancing-Band Benchmark (%)", 0, 20, 5, 1,
                                 help="Kein Trade solange Abweichung < Band (Benchmark)") / 100
            band_mom = st.slider("Rebalancing-Band Momentum (%)", 0, 20, 0, 1,
                                 help="Kein Trade solange Abweichung < Band (Momentum)") / 100
            band_con = st.slider("Rebalancing-Band Contrarian (%)", 0, 20, 0, 1,
                                 help="Kein Trade solange Abweichung < Band (Contrarian)") / 100
            roll_win = st.slider("Rolling-Sharpe Fenster (Jahre)", 1, 3, 1, 1)

        with st.expander("Strategie A: Momentum", expanded=True):
            crash_thr  = st.slider("Crash-Threshold (Z-Score)", -4.0, -0.5, -3.0, 0.1,
                                   help=f"F1 greift wenn Z < Threshold → 20% {ticker1} / 80% {ticker2}")
            target_vol = st.slider("Zielvolatilität F3 (%)", 5, 20, 12, 1,
                                   help="Vol-Scaling-Ziel (annualisiert)") / 100
            zscore_win = st.slider("Z-Score Fenster (Tage)", 21, 504, 21, 1,
                                   help="Rollierendes Fenster für Crash-Z-Score in Handelstagen "
                                        "(21 ≈ 1 Monat, 63 ≈ 1 Quartal, 252 ≈ 1 Jahr)")
            vol_win_d  = st.slider("Vol-Schätzfenster (Tage)", 10, 63, 21, 1)

        with st.expander("Strategie B: Contrarian", expanded=True):
            w_neutral = st.slider("w_neutral (Basisgewicht)", 0.50, 1.00, 0.50, 0.05,
                                  help=f"{ticker1}-Gewicht bei Z=0")
            alpha     = st.slider("α (max. Abweichung)", 0.10, 1.00, 0.50, 0.05,
                                  help="±α Prozentpunkte Bandbreite")
            beta      = st.slider("β (Sensitivität)", 0.10, 2.00, 0.75, 0.05,
                                  help="Steilheit der tanh-Kurve")
            ema_win   = st.slider("EMA-Fenster (Tage)", 5, 63, 21, 1)

        st.markdown("---")
        st.caption(f"tanh-Formel: `w = {w_neutral} + {alpha}·tanh(−{beta}·Z)`")

        with st.expander("tanh-Allokation Vorschau"):
            zv = np.linspace(-4, 4, 300)
            wv = np.clip(w_neutral + alpha * np.tanh(-beta * zv), 0, 1)
            tanh_df = pd.DataFrame({"Z-Score": zv, f"{ticker1}-Gewicht": wv})
            import plotly.express as px
            fig_t = px.line(tanh_df, x="Z-Score", y=f"{ticker1}-Gewicht",
                            title="tanh-Allokation", height=220)
            fig_t.add_hline(y=w_neutral, line_dash="dot", line_color="gray")
            fig_t.update_layout(margin=dict(l=20, r=10, t=40, b=20))
            st.plotly_chart(fig_t, use_container_width=True)

    # ── TITEL ─────────────────────────────────────────────────────────────────
    st.title("Momentum vs. Contrarian — Backtesting Dashboard")
    st.caption(
        f"{ticker1} + {ticker2}  ·  2005–2025  ·  "
        f"Benchmark: {bm_w1*100:.0f}/{(1-bm_w1)*100:.0f} statisch  ·  "
        f"Band BM/Mom/Con: {band_bm*100:.0f}% / {band_mom*100:.0f}% / {band_con*100:.0f}%"
    )

    # ── §10b DATEN & BERECHNUNG ──────────────────────────────────────────────
    # Hier werden die Strategiegewichte und Backtests berechnet.
    # Alle schweren Berechnungen sind gecacht → nur bei Parameteränderung neu.
    with st.spinner("Berechne Strategien …"):
        daily_close, daily_open = load_prices(ticker1, ticker2)
        m_prices = get_monthly(daily_close)

        # Cache-Schlüssel: enthält alle relevanten Parameter inkl. Asset-Ticker
        mom_key = f"{ticker1}_{ticker2}_{crash_thr}_{target_vol}_{zscore_win}_{vol_win_d}"
        con_key = f"{ticker1}_{ticker2}_{w_neutral}_{alpha}_{beta}_{ema_win}"

        w_mom, filter_s, z_mon_s, mom_sig, vs_s, z_mon_s_full = _momentum_weights_cached(
            mom_key, crash_thr, target_vol, zscore_win, vol_win_d, ticker1, ticker2)
        w_con, z_con_d = _contrarian_weights_cached(
            con_key, w_neutral, alpha, beta, ema_win, ticker1, ticker2)

        # w_bm, w_mom, w_con werden UNGESCHNITTEN übergeben – _backtest() schneidet
        # selbst ab start="2005-01-01". Nur so steht der korrekte Dezember-2004-
        # Signalwert bereit, der per T+1 auf alle Januar-2005-Handelstage wirkt.
        # Slice vor dem Aufruf → bfill() würde Januar 2005 mit dem Januar-Signal
        # (das Januar-Daten enthält) füllen → Look-Ahead-Bias.
        w_bm = pd.DataFrame(
            {ticker1: bm_w1, ticker2: 1 - bm_w1},
            index=m_prices.index,
        )

        p_bm  = _backtest(daily_close, daily_open, w_bm,  tc_bps, band=band_bm)
        p_mom = _backtest(daily_close, daily_open, w_mom, tc_bps, band=band_mom)
        p_con = _backtest(daily_close, daily_open, w_con, tc_bps, band=band_con)

        m_bm  = calc_metrics(p_bm,  "Benchmark (80/20)")
        m_mom = calc_metrics(p_mom, "Momentum (A)")
        m_con = calc_metrics(p_con, "Contrarian (B)")

    # ── KPI-TABELLE ───────────────────────────────────────────────────────────
    st.markdown("---")

    def delta_str(val, ref, fmt=".2f", suffix=""):
        d = val - ref
        color = "green" if d > 0 else "red" if d < 0 else "gray"
        sign  = "+" if d >= 0 else ""
        return f"<span style='color:{color};font-size:11px'>{sign}{d:{fmt}}{suffix}</span>"

    def kpi_row(label, bm_val, mom_val, con_val, fmt=".2f", suffix="", higher_better=True):
        """Rendert eine Zeile der KPI-Tabelle."""
        cols = st.columns([2, 2, 2, 2])
        cols[0].markdown(f"<span style='font-size:12px;color:#888'>{label}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<b>{bm_val:{fmt}}{suffix}</b>", unsafe_allow_html=True)

        mom_d = delta_str(mom_val, bm_val, fmt, suffix) if higher_better else delta_str(-mom_val, -bm_val, fmt, suffix)
        con_d = delta_str(con_val, bm_val, fmt, suffix) if higher_better else delta_str(-con_val, -bm_val, fmt, suffix)

        cols[2].markdown(f"<b>{mom_val:{fmt}}{suffix}</b>  {mom_d}", unsafe_allow_html=True)
        cols[3].markdown(f"<b>{con_val:{fmt}}{suffix}</b>  {con_d}", unsafe_allow_html=True)

    # Header
    hcols = st.columns([2, 2, 2, 2])
    hcols[0].markdown("")
    for hc, m, color in zip(hcols[1:], [m_bm, m_mom, m_con], ["#555555", "#1f77b4", "#d62728"]):
        hc.markdown(
            f"<span style='color:{color};font-weight:700;font-size:15px'>&#9646; {m['label']}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

    kpi_row("CAGR",              m_bm["CAGR"]*100,          m_mom["CAGR"]*100,          m_con["CAGR"]*100,          fmt=".2f", suffix="%")
    kpi_row("Volatilität",       m_bm["Vol"]*100,           m_mom["Vol"]*100,           m_con["Vol"]*100,           fmt=".2f", suffix="%", higher_better=False)
    kpi_row("Sharpe Ratio",      m_bm["Sharpe"],            m_mom["Sharpe"],            m_con["Sharpe"],            fmt=".3f")
    kpi_row("Sortino Ratio",     m_bm["Sortino"],           m_mom["Sortino"],           m_con["Sortino"],           fmt=".3f")
    kpi_row("Schiefe (Skew)",    m_bm["Schiefe"],           m_mom["Schiefe"],           m_con["Schiefe"],           fmt=".2f")
    kpi_row("Kurtosis",          m_bm["Kurtosis"],          m_mom["Kurtosis"],          m_con["Kurtosis"],          fmt=".2f", higher_better=False)
    kpi_row("Max Drawdown",      m_bm["Max DD"]*100,        m_mom["Max DD"]*100,        m_con["Max DD"]*100,        fmt=".1f", suffix="%", higher_better=True)
    kpi_row("Worst Month",       m_bm["Worst Month"]*100,   m_mom["Worst Month"]*100,   m_con["Worst Month"]*100,   fmt=".1f", suffix="%", higher_better=True)
    kpi_row("DD Dauer",          m_bm["DD Dauer"],          m_mom["DD Dauer"],          m_con["DD Dauer"],          fmt=".0f", suffix=" Mon.", higher_better=False)
    kpi_row("Calmar Ratio",      m_bm["Calmar"],            m_mom["Calmar"],            m_con["Calmar"],            fmt=".3f")
    kpi_row("Turnover p.a.",     m_bm["Turnover"]*100,      m_mom["Turnover"]*100,      m_con["Turnover"]*100,      fmt=".0f", suffix="%", higher_better=False)
    kpi_row("Kosten-Drag p.a.",  m_bm["Kosten-Drag"]*100,   m_mom["Kosten-Drag"]*100,   m_con["Kosten-Drag"]*100,   fmt=".2f", suffix="%", higher_better=False)

    # ── EQUITY + DRAWDOWN ────────────────────────────────────────────────────
    st.markdown("---")
    ports = {"Benchmark": p_bm, "Momentum": p_mom, "Contrarian": p_con}

    st.plotly_chart(chart_equity(ports), use_container_width=True)
    st.plotly_chart(chart_drawdown(ports), use_container_width=True)

    # ── ASSET-EINZELRENDITEN + VOLATILITÄT ───────────────────────────────────
    st.plotly_chart(chart_asset_vol_return(daily_close, ticker1, ticker2, vol_window=252),
                    use_container_width=True)

    # ── ROLLING SHARPE + FILTER-HISTORY ──────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_rolling_sharpe(ports, window=roll_win * 252),
                        use_container_width=True)
    with c2:
        st.plotly_chart(chart_filter_history(filter_s), use_container_width=True)

    # ── RENDITEVERTEILUNG (KDE, Log-Y, täglich) ────────────────────────────
    st.markdown("---")
    kde_bw = st.slider("KDE Glättung (bw_adjust)", 1.0, 4.0, 1.5, 0.5,
                        help="Höherer Wert → glattere Tails, weniger Rauschen")
    st.plotly_chart(chart_return_distribution(ports, bw_adjust=kde_bw),
                    use_container_width=True)

    # ── ALLOKATION ────────────────────────────────────────────────────────────
    with st.expander("Allokationshistorie", expanded=False):
        w_dict = {
            "Benchmark":  w_bm,
            "Momentum":   w_mom["2005-01-01":],
            "Contrarian": w_con["2005-01-01":],
        }
        st.plotly_chart(chart_allocation(w_dict), use_container_width=True)

    # ── CONTRARIAN Z-SCORE DETAIL ─────────────────────────────────────────────
    with st.expander("Contrarian Z-Score & Allokation (täglich)", expanded=False):
        w_con_daily = (w_con["2005-01-01":].reindex(daily_close["2005-01-01":].index, method="ffill")
                                           .shift(1).bfill()[ticker1])
        st.plotly_chart(chart_contrarian_z(z_con_d["2005-01-01":], w_con_daily, ticker1),
                        use_container_width=True)

    # ── MONATSTABELLE ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Monatstabelle")
    st.caption("Sortierbar · vollständig · exportierbar via Download-Button (CSV)")

    w_mom_d = (w_mom["2005-01-01":].reindex(daily_close["2005-01-01":].index, method="ffill")
                                    .shift(1).bfill()[ticker1])
    w_con_d = (w_con["2005-01-01":].reindex(daily_close["2005-01-01":].index, method="ffill")
                                    .shift(1).bfill()[ticker1])

    mt_key = f"{mom_key}_{con_key}_{tc_bps}_{band_bm}_{band_mom}_{band_con}_{bm_w1}"
    mdf = build_monthly_table(
        mt_key,
        p_bm, p_mom, p_con, w_mom_d, w_con_d,
        filter_s["2005-01-01":], z_mon_s["2005-01-01":], z_con_d,
    )

    # Farbkodierung der Returns
    def color_returns(val):
        if pd.isna(val): return ""
        return f"color: {'green' if val > 0 else 'red'}"

    styled = mdf.style \
        .map(color_returns, subset=["BM %", "Mom %", "α Mom %", "Con %", "α Con %"]) \
        .format(na_rep="—", precision=2)

    st.dataframe(styled, use_container_width=True, height=500)

    # CSV-Download
    csv = mdf.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Monatstabelle als CSV herunterladen",
        csv, "monthly_report.csv", "text/csv",
    )

    # ── REGIME-ANALYSE ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Regime-Analyse")

    regime_data = [
        ("GFC 2007–09",     "2007-10-01", "2009-03-31"),
        ("COVID 2020",      "2020-02-01", "2020-04-30"),
        ("Bärenmarkt 2022", "2022-01-01", "2022-12-31"),
    ]
    rcols = st.columns(len(regime_data))
    for rcol, (rname, rs, re) in zip(rcols, regime_data):
        with rcol:
            st.markdown(f"**{rname}**")
            rows_r = []
            for pname, port in ports.items():
                pv = port["portfolio_value"]
                sl = pv[(pv.index >= rs) & (pv.index <= re)]
                if len(sl) < 2: continue
                ret = (sl.iloc[-1]/sl.iloc[0] - 1) * 100
                dd  = ((sl/sl.cummax())-1).min() * 100
                rows_r.append({"": pname, "Return": f"{ret:+.1f}%", "Max DD": f"{dd:.1f}%"})
            st.table(pd.DataFrame(rows_r).set_index(""))

    # ── KI-ANALYSE EXPORT ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Analyse Export")
    st.caption("Alle Backtestdaten als strukturierter Text")

    if st.button("Bericht generieren", type="primary"):

        # ── Analyzer instanziieren ────────────────────────────────────────────
        analyzer = BacktestReportAnalyzer(ports)

        # ── Parameter-Dict (Reihenfolge = Tabellenreihenfolge) ────────────────
        params = {
            "Asset 1 (Aktien)":            ticker1,
            "Asset 2 (Hedge)":             ticker2,
            "Transaktionskosten":          f"{tc_bps} bps",
            "Rebalancing-Band BM":         f"{band_bm*100:.0f} %",
            "Rebalancing-Band Momentum":   f"{band_mom*100:.0f} %",
            "Rebalancing-Band Contrarian":  f"{band_con*100:.0f} %",
            "Crash-Threshold (Z-Score)":   f"{crash_thr}",
            "Ziel-Volatilität (Mom)":      f"{target_vol*100:.0f} %",
            "Z-Score Fenster (Mom)":       f"{zscore_win} Tage",
            "Vol-Fenster (Mom, tägl.)":    f"{vol_win_d} Tage",
            "w_neutral (Con)":             f"{w_neutral}",
            "α (Con)":                     f"{alpha}",
            "β (Con)":                     f"{beta}",
            "EMA-Fenster (Con, tägl.)":    f"{ema_win} Tage",
        }

        # ── Sensitivitätsanalysen (Backtests mit Parametervariation) ──────────
        with st.spinner("Berechne Sensitivitätsanalyse ..."):

            # 5a. Crash-Threshold 1D (Momentum)
            sens_thr = analyzer.sensitivity_df(
                "Threshold", [-1.0, -2.0, -3.0, -4.0],
                lambda thr: _backtest(daily_close, daily_open,
                    _momentum_weights_cached(
                        f"{ticker1}_{ticker2}_{thr}_{target_vol}_{zscore_win}_{vol_win_d}",
                        thr, target_vol, zscore_win, vol_win_d, ticker1, ticker2)[0],
                    tc_bps, band=band_mom))

            # 5b. Contrarian α × β 2D-Matrix (CAGR)
            sens_ab_cagr = analyzer.sensitivity_2d_df(
                "α", [0.25, 0.50, 0.75],
                "β", [0.25, 0.75, 1.25],
                lambda a, b: _backtest(daily_close, daily_open,
                    _contrarian_weights_cached(
                        f"{ticker1}_{ticker2}_{w_neutral}_{a}_{b}_{ema_win}",
                        w_neutral, a, b, ema_win, ticker1, ticker2)[0],
                    tc_bps, band=band_con),
                metric="CAGR", base_row=alpha, base_col=beta)

            # 5b2. Contrarian α × β 2D-Matrix (Sharpe)
            sens_ab_sharpe = analyzer.sensitivity_2d_df(
                "α", [0.25, 0.50, 0.75],
                "β", [0.25, 0.75, 1.25],
                lambda a, b: _backtest(daily_close, daily_open,
                    _contrarian_weights_cached(
                        f"{ticker1}_{ticker2}_{w_neutral}_{a}_{b}_{ema_win}",
                        w_neutral, a, b, ema_win, ticker1, ticker2)[0],
                    tc_bps, band=band_con),
                metric="Sharpe", base_row=alpha, base_col=beta)

            # 5c. Volatilitätsskalierung: Ziel-Vol × Schätzfenster 2D (CAGR)
            sens_vol_cagr = analyzer.sensitivity_2d_df(
                "Ziel-Vol", [0.08, 0.12, 0.16],
                "Fenster", [10, 21, 63],
                lambda tv, vw: _backtest(daily_close, daily_open,
                    _momentum_weights_cached(
                        f"{ticker1}_{ticker2}_{crash_thr}_{tv}_{zscore_win}_{vw}",
                        crash_thr, tv, zscore_win, vw, ticker1, ticker2)[0],
                    tc_bps, band=band_mom),
                metric="CAGR", base_row=target_vol, base_col=vol_win_d)

            # 5c2. Volatilitätsskalierung: Ziel-Vol × Schätzfenster 2D (Sharpe)
            sens_vol_sharpe = analyzer.sensitivity_2d_df(
                "Ziel-Vol", [0.08, 0.12, 0.16],
                "Fenster", [10, 21, 63],
                lambda tv, vw: _backtest(daily_close, daily_open,
                    _momentum_weights_cached(
                        f"{ticker1}_{ticker2}_{crash_thr}_{tv}_{zscore_win}_{vw}",
                        crash_thr, tv, zscore_win, vw, ticker1, ticker2)[0],
                    tc_bps, band=band_mom),
                metric="Sharpe", base_row=target_vol, base_col=vol_win_d)

            # 5d. TC-Variation (Benchmark + beide Strategien)
            sens_tc = analyzer.sensitivity_tc_df(
                [0, 2, 5, 10, 15, 20, 25, 30, 35],
                lambda c: (_backtest(daily_close, daily_open, w_bm,  c, band=band_bm),
                           _backtest(daily_close, daily_open, w_mom, c, band=band_mom),
                           _backtest(daily_close, daily_open, w_con, c, band=band_con)))

        sensitivity_dfs = {
            "5a. Crash-Threshold-Variation (Momentum)":          sens_thr,
            "5b. Contrarian α × β (CAGR %)":                    sens_ab_cagr,
            "5b2. Contrarian α × β (Sharpe)":                   sens_ab_sharpe,
            "5c. Volatilitätsskalierung Ziel-Vol × Fenster (CAGR %)":  sens_vol_cagr,
            "5c2. Volatilitätsskalierung Ziel-Vol × Fenster (Sharpe)": sens_vol_sharpe,
            "5d. TC-Variation":                                  sens_tc,
        }

        # ── Markdown-Bericht generieren ───────────────────────────────────────
        report_text = analyzer.to_markdown(
            params=params,
            filter_s=filter_s,
            crash_thr=crash_thr,
            sensitivity_dfs=sensitivity_dfs,
            monthly_df=mdf,
        )

        st.text_area(
            "Bericht (alles markieren → Strg+A → Strg+C):",
            value=report_text,
            height=450,
        )

        # JavaScript-Clipboard-Copy
        import streamlit.components.v1 as components
        escaped = report_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        components.html(
            f"""
            <button onclick="navigator.clipboard.writeText(`{escaped}`).then(
                () => this.innerText='Kopiert!',
                () => this.innerText='Fehler beim Kopieren'
            )"
            style="padding:8px 18px;font-size:14px;cursor:pointer;
                   background:#1f77b4;color:white;border:none;border-radius:4px;">
              In Zwischenablage kopieren
            </button>
            """,
            height=50,
        )

    # ── LOGIK-ÜBERPRÜFUNG ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Code überprüfen")
    st.caption("Mathematische Kontrollkalkulationen — prüft ob alle Berechnungen intern konsistent sind")

    if st.button("Logik-Check starten", type="secondary"):
        # ── §10f LOGIK-CHECK: Kontext zusammenstellen und Funktion aufrufen ──
        ctx = dict(
            ports=ports, m_mom=m_mom, m_bm=m_bm, m_con=m_con,
            daily_close=daily_close, daily_open=daily_open,
            w_mom=w_mom, w_con=w_con,
            filter_s=filter_s, z_mon_s=z_mon_s, z_mon_s_full=z_mon_s_full, mom_sig=mom_sig,
            z_con_d=z_con_d, vs_s=vs_s, m_prices=m_prices,
            tc_bps=tc_bps, band_pct=band_mom, band_bm=band_bm, target_vol=target_vol,
            crash_thr=crash_thr, zscore_win=zscore_win, vol_win_d=vol_win_d,
            ema_win=ema_win, w_neutral=w_neutral, alpha=alpha, beta=beta,
            ticker1=ticker1, ticker2=ticker2,
        )

        with st.spinner("Überprüfe …"):
            checks = _run_logic_checks(ctx)

        # Ergebnis anzeigen
        n_ok   = sum(1 for _, ok, _, _w in checks if ok)
        n_warn = sum(1 for _, ok, _, w  in checks if not ok and w)
        n_fail = sum(1 for _, ok, _, w  in checks if not ok and not w)

        col_ok, col_warn, col_fail = st.columns(3)
        col_ok.metric("✓ Bestanden", n_ok)
        col_warn.metric("⚠ Warnungen", n_warn)
        col_fail.metric("✗ Fehler", n_fail)

        for name, ok, detail, warn in checks:
            if ok:
                icon, color = "✓", "green"
            elif warn:
                icon, color = "⚠", "#e67e00"
            else:
                icon, color = "✗", "red"
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{icon}</span> "
                f"<span style='font-size:13px'>{name}</span>"
                + (f"<br><span style='color:#888;font-size:11px;margin-left:18px'>{detail}</span>" if detail else ""),
                unsafe_allow_html=True,
            )

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Bachelorarbeit · Momentum vs. Contrarian · {ticker1} + {ticker2} · 2005–2025  |  "
        "Daten: Yahoo Finance (Adjusted Close)  |  Ausführung: T+1  |  "
        f"TC: {tc_bps} bps/Seite  |  Band BM/Mom/Con: {band_bm*100:.0f}%/{band_mom*100:.0f}%/{band_con*100:.0f}%"
    )


if __name__ == "__main__":
    main()
