# Momentum vs. Contrarian — Backtesting Dashboard

**Bachelorarbeit · Wirtschaftsuniversität Wien · Lukas Kressl**

**Live Demo:** https://bachelor-momentum-contrarian-gkqc677jz6fmyw53d6rypy.streamlit.app/

Empirischer Vergleich zyklischer und antizyklischer Anlagestrategien im Gold-Aktien-Portfolio (SPY + GLD, 2005–2025). Interaktives Streamlit-Dashboard mit Parameter-Sliding, Sensitivitätsanalysen, Regime-Auswertung und automatisierter Selbstvalidierung.

---

## Ergebnisse auf einen Blick

Über 20 Jahre und drei Krisenregimes (GFC, COVID, Bärenmarkt 2022) liefern beide Strategien messbares Alpha gegenüber der statischen 80/20-Benchmark — bei substantiell geringerem Maximum Drawdown.

| Metrik            | Benchmark (80/20) | Momentum    | Contrarian   |
|-------------------|------------------:|------------:|-------------:|
| CAGR              | 11.32 %           | **12.59 %** | **13.59 %**  |
| Sharpe Ratio      | 0.757             | **0.865**   | **0.906**    |
| Sortino Ratio     | 1.075             | 1.232       | 1.315        |
| Max Drawdown      | −43.88 %          | **−26.48 %**| −33.39 %     |
| Volatilität       | 15.84 %           | 15.02 %     | 15.38 %      |
| Turnover p.a.     | 2.8 %             | 441 %       | 377 %        |
| Kosten-Drag p.a.  | 0.00 %            | 0.50 %      | 0.43 %       |

*Zeitraum: 2005–2025 · Transaktionskosten: 5 bps/Seite · Nettorenditen nach TC.*

### Verhalten in Krisenregimes

| Regime           | Benchmark | Momentum | Contrarian |
|------------------|----------:|---------:|-----------:|
| GFC 2007–09      | −35.5 %   | −9.3 %   | −8.7 %     |
| COVID 2020       | −6.2 %    | **+0.5 %** | −2.8 %   |
| Bärenmarkt 2022  | −15.1 %   | −8.1 %   | **+1.6 %** |

Der Momentum-Regime-Filter vermeidet in der GFC rund drei Viertel des Benchmarkverlusts. Der Contrarian nutzt die Korrelationsumkehr zwischen Aktien und Gold im Jahr 2022 und erzielt als einzige der drei Strategien eine positive Jahresrendite.

---

## Strategien

**Strategie A — Momentum (zyklisch).** Sequenzielle Filter-Hierarchie auf Monatsbasis:

1. **F1 Crash-Schutz** — Z-Score < −3.0 → 20/80 defensiv
2. **F2 Absolutes Momentum** — 12-2-Regel nach Jegadeesh negativ → 20/80 defensiv
3. **F3 Volatility Scaling** — sonst: analytische Gewichtung auf 12 % Ziel-Volatilität

Filter-Verteilung im Handelszeitraum: F1 in 0.8 % der Monate aktiv, F2 in 17.1 %, F3 in 82.1 %.

**Strategie B — Contrarian (antizyklisch).** Stetige Allokation über tanh-Funktion auf EMA-basiertem Z-Score:

```
w = clip(0.50 + 0.50 · tanh(−0.75 · Z),  0,  1)
```

Mean-Reversion-Prinzip: negatives Z (Preis unter EMA) → höherer Aktienanteil. Vermeidet durch die stetige Kurve sowohl Schwellenwert-Artefakte als auch binäre 100%-Klumpenrisiken.

---

## Technische Umsetzung

Die Backtest-Engine adressiert drei typische Bias-Quellen:

- **T+1-Ausführungssplit** — Signale aus Monatsende M werden erst am ersten Handelstag von M+1 gehandelt, mit Overnight/Intraday-Split der Tagesrendite. Kein Look-Ahead-Bias.
- **Gewichtsdrift** — tägliche Marktrenditen verschieben die reale Allokation vom Zielgewicht weg. Der Rebalancing-Trigger vergleicht gedriftete Gewichte mit Zielgewichten, nicht den letzten Zielwert mit dem neuen. Auch der statische Benchmark zahlt damit realistische Transaktionskosten.
- **Warmup-Handling** — SPY-Einzeldaten ab 2002 zur Signalberechnung, damit das Z-Score-Fenster zu Handelsstart 2005-01 vollständig befüllt ist.

Selbstvalidierung über 22 Logik-Checks im Dashboard (`_run_logic_checks`, Gruppen A–F) und ein separates CLI-Verifikationsskript (`verification.py`). Alle Performance-Metriken werden in einer zentralen Funktion berechnet (`calc_metrics`), dieselbe Implementierung bedient Haupt-Tabelle, Sensitivitäten und Markdown-Export.

---

## Setup

```bash
# Dependencies installieren
pip install -r requirements.txt

# Dashboard starten
streamlit run dashboard.py

# Optional: Standalone-Validierung im Terminal
python verification.py
```

**Benötigt:** Python 3.10+, Internet-Verbindung beim ersten Start (yfinance-Datendownload, danach per Streamlit-Cache lokal gehalten).

---

## Projektstruktur

```
.
├── dashboard.py          Streamlit-App · Entry-Point
├── verification.py       Standalone-CLI-Validierung (20 Checks)
├── requirements.txt      Package-Versionen
├── KI_Kontext.docx       Methodik-Spezifikation
└── README.md
```

---

## Stack

Python · Streamlit · pandas · NumPy · yfinance · Plotly · SciPy

---

## Hinweis

Der Code entsteht im Rahmen einer Bachelorarbeit und dient akademischen Zwecken. Keine Anlageempfehlung. Vergangene Wertentwicklung ist kein Indikator für zukünftige Ergebnisse.
