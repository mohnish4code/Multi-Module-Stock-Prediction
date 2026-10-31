.PHONY: install lint test news backtest backtest-news tune train api ui up down

PY ?= python

install:
	$(PY) -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	MLFLOW_DISABLE=1 pytest

news:            ## backfill historical headlines (slow, resumable)
	$(PY) -m src.data.news

backtest:        ## walk-forward backtest, price features
	$(PY) backtest.py

backtest-news:   ## same models with FinBERT news features
	$(PY) backtest.py --features news --models xgboost lstm transformer

tune:
	$(PY) tune.py --model xgboost --trials 60
	$(PY) tune.py --model lstm --trials 20
	$(PY) tune.py --model transformer --trials 15

train:           ## fit serving models on all history
	$(PY) train.py

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

up:
	docker compose up --build -d

down:
	docker compose down
