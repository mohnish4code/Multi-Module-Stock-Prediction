"""Optuna hyperparameter search for xgboost / lstm / transformer.

    python tune.py --model xgboost --trials 60
    python tune.py --model transformer --trials 15 --tickers TCS HDFCBANK RELIANCE

No test data is used: every trial fits on fold 0's training rows and is scored on
fold 0's inner validation rows, which end before the first test day of the
walk-forward backtest. The objective is validation MSE / Var(y_val) (= 1 - R^2),
averaged over the tuning tickers, so it is comparable across models.
Best parameters are merged into configs/tuned.yaml; every trial is logged to MLflow.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import optuna
import yaml

from src.config import TUNED_CONFIG, load_config, resolve_ticker
from src.evaluation.backtest import load_dataset, make_folds
from src.logging_setup import setup_logging
from src.models.dataset import Dataset
from src.models.registry import get_model_class
from src.tracking import run

logger = logging.getLogger("tune")

DEFAULT_TICKERS = ["TCS", "HDFCBANK", "RELIANCE"]  # IT, bank, conglomerate
# Values the hand-set models effectively use for searched params that are not in config.yaml.
IMPLICIT_DEFAULTS = {
    "xgboost": {"min_child_weight": 1.0, "reg_lambda": 1.0},
    "lstm": {"num_layers": 1, "weight_decay": 1e-6},
    "transformer": {},
}


def search_space(model: str, trial: optuna.Trial) -> dict[str, Any]:
    if model == "xgboost":
        return {
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
    if model == "lstm":
        return {
            "units": trial.suggest_categorical("units", [16, 24, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "dense_units": trial.suggest_categorical("dense_units", [8, 16]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": 120,
            "patience": 15,
        }
    if model == "transformer":
        d_model = trial.suggest_categorical("d_model", [16, 32, 64])
        return {
            "d_model": d_model,
            "nhead": trial.suggest_categorical("nhead", [2, 4]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "dense_units": 16,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": 80,
            "patience": 10,
        }
    raise ValueError(f"No search space for '{model}'")


def validation_score(model_name: str, params: dict[str, Any], ds: Dataset, cfg: dict[str, Any]) -> float:
    fold = make_folds(cfg, len(ds.y))[0]
    model = get_model_class(model_name)(params, cfg["features"]["sequence_length"], cfg["backtest"]["seed"])
    model.fit(ds, fold.train, fold.val)
    y = ds.y[fold.val]
    return float(np.mean((model.predict(ds, fold.val) - y) ** 2) / np.var(y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["xgboost", "lstm", "transformer"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    setup_logging()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(use_tuned=False)
    datasets = [load_dataset(cfg, resolve_ticker(cfg, t)) for t in args.tickers]
    baseline = dict(cfg["models"][args.model])

    with run("tuning", f"{args.model} x{args.trials}") as parent:
        parent.params({"model": args.model, "trials": args.trials, "tickers": ",".join(args.tickers)})

        def objective(trial: optuna.Trial) -> float:
            params = search_space(args.model, trial)
            scores = []
            for i, ds in enumerate(datasets):
                scores.append(validation_score(args.model, params, ds, cfg))
                trial.report(float(np.mean(scores)), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            value = float(np.mean(scores))
            with run("tuning", f"{args.model} trial {trial.number}", nested=True) as r:
                r.params(params)
                r.metrics({"val_mse_over_var": value})
            logger.info("trial %d: %.5f  %s", trial.number, value, trial.params)
            return value

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        # Trial 0 is the hand-set configuration, so the study reports a like-for-like improvement.
        study.enqueue_trial({**IMPLICIT_DEFAULTS[args.model], **baseline})
        study.optimize(objective, n_trials=args.trials)

        best = search_space(args.model, optuna.trial.FixedTrial(study.best_params))
        baseline_value = study.trials[0].value
        parent.metrics({"best_val_mse_over_var": study.best_value,
                        "baseline_val_mse_over_var": baseline_value or float("nan")})
        parent.params({f"best.{k}": v for k, v in best.items()})

    tuned = yaml.safe_load(TUNED_CONFIG.read_text(encoding="utf-8")) if TUNED_CONFIG.exists() else {}
    tuned = tuned or {}
    tuned.setdefault("models", {})[args.model] = best
    tuned.setdefault("studies", {})[args.model] = {
        "trials": len(study.trials),
        "tickers": args.tickers,
        "objective": "fold-0 validation MSE / Var(y), mean over tickers",
        "best_value": round(study.best_value, 6),
        "hand_set_value": round(baseline_value, 6) if baseline_value is not None else None,
    }
    TUNED_CONFIG.write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    logger.info("best %.5f (hand-set %.5f) -> %s\n%s", study.best_value, baseline_value or float("nan"),
                TUNED_CONFIG, best)


if __name__ == "__main__":
    main()
