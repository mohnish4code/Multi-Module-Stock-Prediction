# =========================================================
# MULTI-MODULE STOCK PREDICTION SYSTEM
# COMMAND-LINE RUNNER
#
# Thin CLI over the exact same shared pipeline the
# Streamlit app uses (modules/inference.py), so the two
# entry points can never disagree.
# =========================================================

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from modules.data_collection import get_stock_data
from modules.technical_analysis import analyze_technical_indicators
from modules.inference import (
    load_artifacts,
    load_metrics,
    predict_next_day,
)
from modules.news_sentiment import analyze_company_sentiment
from modules.coordination_engine import coordinate_modules
from modules.confidence_engine import calculate_confidence
from modules.risk_engine import calculate_risk
from modules.decision_engine import make_final_decision

from config import (
    COMPANIES,
    SEQUENCE_LENGTH,
    LIVE_DATA_PERIOD,
)


def _normalise_sentiment(label):
    label = str(label).upper().strip()
    if label in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return label
    return "NEUTRAL"


def run(company_input=None):

    # ---------------- choose company ----------------
    items = list(COMPANIES.items())

    if company_input is None:
        print("\n" + "=" * 70)
        print("MULTI-MODULE STOCK PREDICTION SYSTEM")
        print("=" * 70 + "\nAVAILABLE COMPANIES:\n")
        for i, (t, info) in enumerate(items, 1):
            print(f"{i:>2}. {info['name']} ({t})")

        while True:
            try:
                choice = int(input("\nEnter company number: "))
                if 1 <= choice <= len(items):
                    break
            except ValueError:
                pass
            print("Please enter a valid number.")
        ticker, info = items[choice - 1]
    else:
        key = str(company_input).upper().strip()
        match = [
            (t, i) for (t, i) in items
            if key in (t, t.replace(".NS", ""), i["model_folder"].upper())
        ]
        if not match:
            print(f"Company '{company_input}' not found.")
            return
        ticker, info = match[0]

    name = info["name"]
    model_folder = info["model_folder"]

    print("\n" + "=" * 70)
    print(f"SELECTED: {name}  ({ticker})")
    print("=" * 70)

    # ---------------- model + data ----------------
    print("\nLoading model artifacts...")
    artifacts = load_artifacts(model_folder)
    metrics = load_metrics(model_folder)

    print("Fetching latest market data...")
    df = get_stock_data(ticker=ticker, period=LIVE_DATA_PERIOD, interval="1d")
    if df is None or df.empty:
        print("No stock data was received.")
        return

    # ---------------- LSTM forecast (shared path) ----------------
    print("Generating next-day forecast...")
    pred = predict_next_day(df, artifacts, sequence_length=SEQUENCE_LENGTH)

    current_price = pred["current_price"]
    predicted_price = pred["predicted_price"]
    price_change = pred["predicted_change"]
    pct_change = pred["predicted_change_percent"]
    prediction_direction = pred["prediction_direction"]
    recommendation_signal = pred["recommendation_signal"]

    # ---------------- technical analysis ----------------
    print("Running technical analysis...")
    technical_result = analyze_technical_indicators(pred["df_features"])
    technical_signal = technical_result["technical_signal"]

    # ---------------- news sentiment ----------------
    print("Fetching and analysing news...")
    news_result = analyze_company_sentiment(company_name=name, ticker=ticker)
    news_sentiment = _normalise_sentiment(news_result.get("sentiment_label", "NEUTRAL"))

    # ---------------- coordination / confidence / risk ----------------
    coordination_result = coordinate_modules(
        prediction_direction, technical_signal, news_sentiment
    )
    agreement = coordination_result["agreement"]

    confidence_result = calculate_confidence(
        prediction_direction, technical_signal, news_sentiment, agreement
    )
    confidence_score = confidence_result["confidence_score"]

    risk_result = calculate_risk(
        volatility=technical_result["volatility_level"],
        predicted_change_percent=pct_change,
        technical_signal=technical_signal,
        confidence_score=confidence_score,
        agreement=agreement,
    )
    risk_level = risk_result["risk_level"]

    decision_result = make_final_decision(
        coordinated_signal=coordination_result["coordinated_signal"],
        agreement=agreement,
        confidence_score=confidence_score,
        risk_level=risk_level,
        predicted_change_percent=pct_change,
        technical_signal=technical_signal,
        news_sentiment=news_sentiment,
    )

    # ---------------- report ----------------
    line = "-" * 70
    print("\n" + "=" * 70)
    print(f"RESULT  |  {name}  ({ticker})")
    print("=" * 70)

    print("\nMARKET DATA")
    print(line)
    print(f"Current price          Rs {current_price:,.2f}")
    print(f"Predicted next close   Rs {predicted_price:,.2f}")
    print(f"Expected change        Rs {price_change:+,.2f}  ({pct_change:+.2f}%)")
    print(f"Predicted return       {pred['predicted_return'] * 100:+.2f}%")

    if metrics:
        print("\nMODEL ACCURACY (hold-out back-test)")
        print(line)
        print(
            f"High-conviction accuracy  {metrics.get('conviction_accuracy')}% "
            f"(top {metrics.get('conviction_coverage')}% of days)"
        )
        print(
            f"All-days direction        {metrics.get('directional_accuracy')}% "
            f"(naive baseline {metrics.get('baseline_always_up')}%)"
        )
        print(f"Price error (MAPE)        {metrics.get('mape')}%")
        print(f"Back-test days            {metrics.get('n_samples')}")
        print(f"Trained                   {metrics.get('trained_at', 'unknown')}")

    print("\nMODULE SIGNALS")
    print(line)
    print(f"1. LSTM forecast      {prediction_direction}  ->  {recommendation_signal}")
    print(f"2. Technical          {technical_signal}")
    print(f"3. News sentiment     {news_sentiment}")
    print(
        f"   feed fetched {news_result.get('fetched_at', 'n/a')} | "
        f"most recent {news_result.get('most_recent', 'n/a')} | "
        f"{news_result.get('total_news', 0)} headlines"
    )

    print("\nCOORDINATION")
    print(line)
    print(f"Coordinated signal   {coordination_result['coordinated_signal']}")
    print(f"Agreement            {agreement}")
    print(f"Confidence           {confidence_score}%  ({confidence_result['confidence_level']})")
    print(f"Risk                 {risk_result['risk_score']}/10  ({risk_level})")

    print("\nFINAL DECISION")
    print(line)
    print(f">>> {decision_result['recommendation']} <<<")
    print("\n" + decision_result["explanation"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
