# AlphaStream — Multi-Module Stock Prediction System

An AI equity-research terminal for 10 NSE large caps. It combines three
independent signal modules and reconciles them into a single next-day call,
served through a trading-terminal style Streamlit UI.

```
LSTM return forecast ─┐
Technical analysis  ──┼─► coordination ─► confidence ─► risk ─► decision
Live news sentiment ──┘
```

## What each part does

| Module | File | Output |
|---|---|---|
| LSTM forecast | `modules/inference.py`, `models/model.py` | next-day **log return** → price, BULLISH/BEARISH/NEUTRAL |
| Technical analysis | `modules/technical_analysis.py` | RSI / MACD / SMA / Bollinger read → BULLISH/BEARISH/NEUTRAL |
| News sentiment | `modules/news_collection.py`, `modules/news_sentiment.py` | live Google News RSS + keyword scoring → POSITIVE/NEUTRAL/NEGATIVE |
| Coordination | `modules/coordination_engine.py` | 3-way vote + agreement level |
| Confidence | `modules/confidence_engine.py` | 0–100 score |
| Risk | `modules/risk_engine.py` | 0–10 score + factors |
| Decision | `modules/recommendation_engine.py` (app) / `modules/decision_engine.py` (CLI) | BUY / HOLD / SELL + explanation |

All three entry points — the app, `system.py`, and `modules/predictions.py` —
route through the single shared pipeline in `modules/inference.py`, so they
cannot disagree.

## The model

Per-ticker LSTM trained on **stationary features** (returns, indicator ratios,
momentum, distance from 52-week high/low, day-of-week) predicting the
**next-day log return**, not the absolute price. This is deliberate: a price-level
model with a `MinMaxScaler` fitted on old data goes out of range as prices rise
and is forced to extrapolate. Returns stay in a stable band for the life of the
model.

Each `models/<TICKER>/` folder holds:

```
lstm_model.keras      feature_scaler.pkl    target_scaler.pkl
metrics.json          training_info.txt
```

### Accuracy (hold-out back-test, 356 unseen days per stock)

| Metric | Typical |
|---|---|
| Price error (MAPE) | **~1%** |
| High-conviction direction (top⅓ days) | 48% avg (39–55% by stock) |
| All-days direction | ~48% |

Honest read: the **price-magnitude** forecast is solid; next-day **direction**
is near coin-flip and has no reliable edge (this is the nature of daily equity
returns, not a tuning gap). The app shows the real back-test numbers with naive
baselines so the signal is never oversold.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Usage

```bash
# trading terminal UI
streamlit run app.py

# one-shot CLI report
python system.py TCS

# just the price prediction
python scripts/predict_stock.py RELIANCE

# just the news sentiment
python scripts/analyse_sentiment.py INFY
```

## Retraining

```bash
python training/retrain_all.py            # all 10 (~5–6 min)
python training/retrain_all.py TCS INFY   # subset
python training/train_model.py TCS        # single company
python training/evaluate_model.py TCS     # recompute metrics.json only
```

Training settings live in `config.py` (`EPOCHS`, `BATCH_SIZE`, `TRAIN_RATIO`,
`SEQUENCE_LENGTH`, `NEUTRAL_THRESHOLD_PCT`, news windows).

## Disclaimer

Research / education tool. Forecasts are model output, not investment advice.
