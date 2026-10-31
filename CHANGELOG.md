# Changelog

## 2.0.0

A rewrite focused on measuring the models honestly and making the project deployable.

### Evaluation
- Walk-forward backtest: expanding training window, 6 one-year test folds (Sep 2020 – Oct 2026),
  every model refit per fold. A 5-row embargo separates train / validation / test so no label leaks
  across a boundary; scalers are fit on training rows only.
- Baselines: random walk, persistence, majority direction, ARIMA (order by AIC, one-step-ahead
  without refitting).
- Metrics: RMSE / MAE / MAPE, directional accuracy with binomial p-values, and a long/flat strategy
  net of 0.12% per side (Sharpe, CAGR, max drawdown, time in market) against buy & hold.
- Leakage checks: shuffled targets score ~50%, a deliberately leaked target scores ~100%, and tests
  assert that appending future bars never changes past features.

### Models
- PyTorch LSTM (port of the original Keras model) and a Transformer encoder, behind the same
  `Forecaster` interface as XGBoost and the baselines.
- The Keras → PyTorch port was checked fold-for-fold before TensorFlow was dropped: same
  architecture, mean direction accuracy 50.2% vs 49.6% and net Sharpe 0.36 vs 0.12 (both within
  noise). The two versions' daily forecasts correlate only 0.15–0.55, which says more about how
  little signal there is than about either framework.
- Optuna tuning on the first fold's validation block only (objective: validation MSE / Var(y)).
  Results are stored in `configs/tuned.yaml` and picked up automatically.

### News sentiment
- Weekly Google News backfill per company since 2011 (`python -m src.data.news`), about 63,600 unique headlines.
- FinBERT (`ProsusAI/finbert`) headline scores turned into four daily features. A headline only
  affects trading days strictly after its IST publication date, because older RSS timestamps are often
  date-only.
- With/without-news ablation for XGBoost and the LSTM on identical folds. The Transformer was left out
  of the ablation to keep the run to a few hours on CPU.
- The live news panel uses FinBERT, falling back to the keyword lexicon if the model is unavailable.

### Serving
- FastAPI service (`/predict`, `/backtest`, `/tickers`, `/quotes`, `/health`) with Pydantic models,
  per-IP rate limiting and JSON errors (404 unknown ticker, 503 model not trained, 422 bad data).
- Streamlit app talks to the API only; new tab with the walk-forward equity curve, baseline table and
  per-fold accuracy.
- Docker images for api / ui / mlflow and a compose file; GitHub Actions runs ruff, pytest and the
  three Docker builds.

### Project structure
- Code moved into `src/` (data, features, models, evaluation, signals, api); one config file
  (`configs/config.yaml`); `train.py`, `backtest.py` and `tune.py` loop over all tickers.
- One decision engine shared by the CLI, API and UI (previously the app and the CLI used different
  rules and could disagree).
- Pinned requirements, type hints, logging instead of prints, MLflow tracking, 77 offline tests.

### Fixes found along the way
- The old evaluation script rebuilt the test set with a freshly fitted scaler instead of the saved
  one, and inverted targets with a different scaler than the one that scaled them.
- The old 1% "neutral" band meant the LSTM signal was NEUTRAL on every single backtest day; the
  band is now 0.10%, in line with the size of the forecasts.
- Yahoo sometimes returns the latest session with volume but no prices; such bars are now dropped.

## 1.0.0

First prototype: a Keras LSTM per stock trained on a single 70/15/15 split, technical indicators,
keyword news sentiment and a Streamlit dashboard.
