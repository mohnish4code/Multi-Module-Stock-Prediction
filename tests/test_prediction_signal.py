import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    PROJECT_ROOT
)


from modules.prediction_signal import (
    generate_prediction_signal
)


# -------------------------------------------------
# TEST CASES
# -------------------------------------------------

test_cases = [

    {
        "current_price": 1000,
        "predicted_price": 1050
    },

    {
        "current_price": 1000,
        "predicted_price": 1005
    },

    {
        "current_price": 1000,
        "predicted_price": 950
    }

]


print("\n" + "=" * 60)

print(
    "PREDICTION SIGNAL TEST"
)

print("=" * 60)


for case in test_cases:

    result = generate_prediction_signal(

        current_price=
        case["current_price"],

        predicted_price=
        case["predicted_price"],

        threshold=1.0

    )


    print("\n" + "-" * 60)


    print(

        "Current Price:",

        result["current_price"]

    )


    print(

        "Predicted Price:",

        result["predicted_price"]

    )


    print(

        "Predicted Change:",

        f"{result['predicted_change_percent']:.2f}%"

    )


    print(

        "Prediction Direction:",

        result["prediction_direction"]

    )


    print(

        "Recommendation Signal:",

        result["recommendation_signal"]

    )


print("\n" + "=" * 60)

print(
    "PREDICTION SIGNAL TEST COMPLETED"
)

print("=" * 60)