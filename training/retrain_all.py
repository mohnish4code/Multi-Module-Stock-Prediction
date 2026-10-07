import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from config import COMPANIES
from training.train_model import train_company


def main(only=None):
    """
    Retrains every company in config.COMPANIES with the
    current stationary returns pipeline.

    `only` : optional list of model folders / tickers to
             restrict the run.
    """

    targets = list(COMPANIES.items())

    if only:
        only_up = {o.upper().replace(".NS", "") for o in only}
        targets = [
            (t, i) for (t, i) in targets
            if i["model_folder"].upper() in only_up
            or t.replace(".NS", "") in only_up
        ]

    results = []
    started = time.time()

    for idx, (ticker, info) in enumerate(targets, 1):

        print(f"\n########## [{idx}/{len(targets)}] {info['name']} ##########")

        try:
            m = train_company(ticker, verbose=2)
            results.append((info["model_folder"], m, None))
        except Exception as exc:  # keep going on failure
            traceback.print_exc()
            results.append((info["model_folder"], None, str(exc)))

    # ---------------- summary ----------------
    elapsed = time.time() - started

    print("\n" + "=" * 72)
    print("RETRAIN SUMMARY")
    print("=" * 72)
    print(f"{'MODEL':<12}{'DIR-ACC':>9}{'BASE':>8}{'CONVICT':>9}{'MAPE':>8}  STATUS")
    print("-" * 72)

    for folder, m, err in results:
        if m is None:
            print(f"{folder:<12}{'-':>9}{'-':>8}{'-':>9}{'-':>8}  FAILED: {err[:24]}")
        else:
            conv = m.get("conviction_accuracy")
            print(
                f"{folder:<12}"
                f"{m['directional_accuracy']:>8.2f}%"
                f"{m['baseline_always_up']:>7.1f}%"
                f"{(conv if conv is not None else 0):>8.2f}%"
                f"{m['mape']:>7.2f}%"
                f"  ok"
            )

    ok = [m for _, m, e in results if m is not None]
    if ok:
        avg_dir = sum(m["directional_accuracy"] for m in ok) / len(ok)
        avg_conv = sum((m.get("conviction_accuracy") or 0) for m in ok) / len(ok)
        avg_mape = sum(m["mape"] for m in ok) / len(ok)
        print("-" * 72)
        print(
            f"{'AVERAGE':<12}{avg_dir:>8.2f}%{'':>8}"
            f"{avg_conv:>8.2f}%{avg_mape:>7.2f}%"
        )

    print(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only)
