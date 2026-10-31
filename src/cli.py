"""Command-line report for one ticker.

    python -m src.cli TCS
    python -m src.cli INFY --no-news
"""

from __future__ import annotations

import argparse
import logging

from src.config import load_config, resolve_ticker
from src.logging_setup import setup_logging
from src.pipeline import analyse

logger = logging.getLogger("cli")


def format_report(a: dict) -> str:
    f, t, n, d = a["forecast"], a["technical"], a["news"], a["decision"]
    bt = (n.get("news") or [])
    lines = [
        f"{a['name']} ({a['ticker']})  as of {f['as_of']}",
        f"  Forecast [{f['model']}]  Rs {f['current_price']:,.2f} -> Rs {f['predicted_price']:,.2f} "
        f"({f['predicted_change_pct']:+.3f}%)  {f['direction']}",
        f"  Technical               {t['technical_signal']}  (RSI {t['rsi']:.1f}, trend {t['trend']}, "
        f"MACD {t['macd_trend']}, vol {t['volatility_level']})",
        f"  News [{n['method']}]        {n['sentiment_label']}  ({n['total_news']} headlines, "
        f"score {n['sentiment_score']:+.2f})",
        f"  Coordinated             {d['coordinated_signal']}  ({d['agreement']} agreement)",
        f"  Agreement / risk        {d['confidence_score']}/100 ({d['confidence_level']}), "
        f"risk {d['risk_score']}/10 ({d['risk_level']})",
        f"  >>> {d['recommendation']} <<<",
        f"  {d['explanation']}",
    ]
    lines += [f"    - {h['title']}  [{h['sentiment']}]" for h in bt[:5]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("--no-news", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    ticker = resolve_ticker(cfg, args.ticker)
    logger.info("\n%s", format_report(analyse(ticker, cfg, include_news=not args.no_news)))


if __name__ == "__main__":
    main()
