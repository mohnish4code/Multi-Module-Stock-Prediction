.PHONY: install lint test news backtest backtest-news tune train ui api dev up down

PY ?= python

install:
	$(PY) -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt
	cd frontend && npm ci

lint:
	ruff check .

test:
	MLFLOW_DISABLE=1 pytest
	cd frontend && npm test

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

ui:              ## build the React app; the API then serves it at http://localhost:8000
	cd frontend && npm run build

api:
	uvicorn src.api.main:app --port 8000

dev:             ## UI with hot reload on :5173 (run `make api` in another terminal)
	cd frontend && npm run dev

up:
	docker compose up --build -d

down:
	docker compose down
