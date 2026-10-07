import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COMPANIES, SEQUENCE_LENGTH, LIVE_DATA_PERIOD
from modules.data_collection import get_stock_data
from modules.inference import load_artifacts, load_metrics, predict_next_day


print("\n" + "=" * 70)
print("TESTING ALL TRAINED MODELS (shared inference path)")
print("=" * 70)

ok, failed = [], []

for ticker, info in COMPANIES.items():

    name = info["name"]
    folder = info["model_folder"]

    print("\n" + "-" * 70)
    print(f"{name}  ({ticker})")
    print("-" * 70)

    try:
        artifacts = load_artifacts(folder)

        df = get_stock_data(ticker=ticker, period=LIVE_DATA_PERIOD, interval="1d")
        if df is None or df.empty:
            raise RuntimeError("no market data")

        pred = predict_next_day(df, artifacts, sequence_length=SEQUENCE_LENGTH)
        metrics = load_metrics(folder)

        print(
            f"OK  current Rs {pred['current_price']:,.2f}  "
            f"-> predicted Rs {pred['predicted_price']:,.2f}  "
            f"({pred['predicted_change_percent']:+.2f}%)  "
            f"{pred['prediction_direction']}"
        )
        if metrics:
            print(
                f"    accuracy: conviction {metrics.get('conviction_accuracy')}%  "
                f"all-days {metrics.get('directional_accuracy')}%  "
                f"MAPE {metrics.get('mape')}%"
            )
        else:
            print("    WARNING: metrics.json missing")

        ok.append(name)

    except Exception as exc:
        print(f"FAILED: {exc}")
        failed.append((name, str(exc)))


print("\n" + "=" * 70)
print(f"PASSED {len(ok)}/{len(COMPANIES)}")
for n, e in failed:
    print(f"  FAIL {n}: {e}")
print("=" * 70)
