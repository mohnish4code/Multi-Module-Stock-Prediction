import datetime
import socket
import urllib.request
from urllib.parse import quote

import feedparser

try:
    from config import (
        NEWS_RECENCY_DAYS,
        NEWS_MIN_ITEMS,
        NEWS_MAX_ITEMS,
    )
except Exception:
    NEWS_RECENCY_DAYS = 21
    NEWS_MIN_ITEMS = 4
    NEWS_MAX_ITEMS = 15


# =================================================
# COMPANY SEARCH QUERIES  (keyed by model folder)
# =================================================

COMPANY_NEWS_QUERIES = {
    "TCS": "Tata Consultancy Services TCS India stock",
    "RELIANCE": "Reliance Industries India stock",
    "INFY": "Infosys India stock",
    "HDFCBANK": "HDFC Bank India stock",
    "ICICIBANK": "ICICI Bank India stock",
    "SBIN": "State Bank of India SBI stock",
    "BHARTIARTL": "Bharti Airtel India stock",
    "LT": "Larsen & Toubro L&T India stock",
    "HINDUNILVR": "Hindustan Unilever HUL India stock",
    "ITC": "ITC Limited India stock",
}


# Full display name (from config) -> short key above.
COMPANY_NAME_TO_KEY = {
    "TATA CONSULTANCY SERVICES": "TCS",
    "RELIANCE INDUSTRIES": "RELIANCE",
    "INFOSYS": "INFY",
    "HDFC BANK": "HDFCBANK",
    "ICICI BANK": "ICICIBANK",
    "STATE BANK OF INDIA": "SBIN",
    "BHARTI AIRTEL": "BHARTIARTL",
    "LARSEN & TOUBRO": "LT",
    "HINDUSTAN UNILEVER": "HINDUNILVR",
    "ITC": "ITC",
}


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _resolve_key(company_name):
    key = str(company_name).upper().strip().replace(".NS", "")
    return COMPANY_NAME_TO_KEY.get(key, key)


def _entry_datetime(entry):
    """feedparser struct_time -> aware UTC datetime, or None."""
    for attr in ("published_parsed", "updated_parsed"):
        st = entry.get(attr)
        if st:
            try:
                return datetime.datetime(*st[:6], tzinfo=datetime.timezone.utc)
            except Exception:
                pass
    return None


def _fetch_feed(url, timeout=10):
    """Fetch the RSS bytes ourselves so we get a real
    timeout and a browser UA (Google News blocks the
    default urllib agent intermittently)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return feedparser.parse(raw)
    except (urllib.error.URLError, socket.timeout, Exception) as exc:
        print(f"News fetch failed ({exc}); retrying via feedparser directly...")
        try:
            return feedparser.parse(url, agent=_USER_AGENT)
        except Exception as exc2:
            print(f"News fetch failed again: {exc2}")
            return None


# -------------------------------------------------
# GET COMPANY NEWS  (real-time, newest first)
# -------------------------------------------------

def get_stock_news(company_name, max_news=None, recency_days=None,
                   min_items=None):
    """
    Fetches company news from Google News RSS.

    Behaviour (per the product spec): keep the freshest
    headlines available. Items newer than `recency_days`
    are always kept; if that yields fewer than
    `min_items`, we top up with the next-newest items
    regardless of age so a thinly-covered stock still
    gets analysed.

    Every item carries `published` (datetime|None) and
    `published_str`. The list is sorted newest-first.
    Also returns nothing extra here - see
    news_sentiment.analyze_company_sentiment for the
    fetched_at / most_recent envelope.
    """

    max_news = max_news or NEWS_MAX_ITEMS
    recency_days = NEWS_RECENCY_DAYS if recency_days is None else recency_days
    min_items = NEWS_MIN_ITEMS if min_items is None else min_items

    key = _resolve_key(company_name)
    search_query = COMPANY_NEWS_QUERIES.get(key, f"{company_name} India stock")

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={quote(search_query)}+when:1y"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = _fetch_feed(rss_url)

    if feed is None or not getattr(feed, "entries", None):
        print("No news entries returned.")
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    items = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        published = _entry_datetime(entry)

        source = entry.get("source", {})
        publisher = (
            source.get("title", "Unknown")
            if isinstance(source, dict) else "Unknown"
        )

        age_days = (
            (now - published).total_seconds() / 86400.0
            if published is not None else None
        )

        items.append({
            "title": title,
            "publisher": publisher,
            "link": entry.get("link", ""),
            "published": published,
            "published_str": (
                published.astimezone().strftime("%d %b %Y, %H:%M")
                if published is not None else "date unknown"
            ),
            "age_days": round(age_days, 1) if age_days is not None else None,
        })

    # newest first; unknown dates sink to the bottom
    items.sort(
        key=lambda it: it["published"] or datetime.datetime.min.replace(
            tzinfo=datetime.timezone.utc
        ),
        reverse=True,
    )

    fresh = [
        it for it in items
        if it["age_days"] is not None and it["age_days"] <= recency_days
    ]

    if len(fresh) >= min_items:
        selected = fresh
    else:
        # not enough fresh news -> take the newest available
        selected = items

    return selected[:max_news]
