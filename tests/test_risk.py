import os
import sys


# =================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    PROJECT_ROOT
)


# =================================================
# IMPORT RISK ENGINE
# =================================================

from modules.risk_engine import (
    calculate_risk
)


# =================================================
# TEST DATA
# =================================================

result = calculate_risk(

    volatility="MEDIUM",

    predicted_change_percent=-2.40,

    technical_signal="NEUTRAL",

    confidence_score=60,

    agreement="MIXED"
)


# =================================================
# DISPLAY RESULTS
# =================================================

print(
    "\n" + "=" * 50
)

print(
    "RISK ANALYSIS"
)

print(
    "=" * 50
)


print(
    f"\nRisk Score: "
    f"{result['risk_score']}/10"
)


print(
    f"Risk Level: "
    f"{result['risk_level']}"
)


print(
    "\nRISK FACTORS:\n"
)


for factor in result["risk_factors"]:

    print(
        f"- {factor}"
    )


print(
    "\n" + "=" * 50
)