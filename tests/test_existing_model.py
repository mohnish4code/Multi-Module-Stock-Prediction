import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import SEQUENCE_LENGTH, LIVE_DATA_PERIOD
from modules.data_collection import get_stock_data
from modules.inference import load_artifacts, load_metrics, predict_next_day

TICKER = "TCS.NS"
FOLDER = "TCS"

print("\n" + "=" * 60)
print(f"EXISTING MODEL TEST  ({TICKER})")
print("=" * 60)

artifacts = load_artifacts(FOLDER)

df = get_stock_data(ticker=TICKER, period=LIVE_DATA_PERIOD, interval="1d")

pred = predict_next_day(df, artifacts, sequence_length=SEQUENCE_LENGTH)
metrics = load_metrics(FOLDER)

print(f"\nCurrent price      Rs {pred['current_price']:,.2f}")
print(f"Predicted close    Rs {pred['predicted_price']:,.2f}")
print(f"Expected change    {pred['predicted_change_percent']:+.2f}%")
print(f"Predicted return   {pred['predicted_return'] * 100:+.3f}%")
print(f"Direction          {pred['prediction_direction']} -> {pred['recommendation_signal']}")

if metrics:
    print(f"\nBack-test conviction accuracy  {metrics.get('conviction_accuracy')}%")
    print(f"Back-test all-days accuracy    {metrics.get('directional_accuracy')}%")
    print(f"Back-test MAPE                 {metrics.get('mape')}%")

print("\n" + "=" * 60)
print("PASSED")
print("=" * 60)
