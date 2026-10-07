import yfinance as yf


# =================================================
# BATCH QUOTES  (for the watchlist strip)
# =================================================

def get_quotes(tickers):
    """
    One network call for every ticker. Returns:
        { ticker: {"price": float, "prev": float,
                   "change": float, "change_pct": float} }
    Missing tickers are simply omitted.
    """

    if not tickers:
        return {}

    out = {}

    try:
        data = yf.download(
            list(tickers),
            period="7d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"quote fetch failed: {exc}")
        return {}

    for t in tickers:
        try:
            if len(tickers) == 1:
                closes = data["Close"].dropna()
            else:
                closes = data[t]["Close"].dropna()

            if len(closes) < 2:
                continue

            price = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change = price - prev

            out[t] = {
                "price": price,
                "prev": prev,
                "change": change,
                "change_pct": (change / prev) * 100.0 if prev else 0.0,
            }
        except Exception:  # noqa: BLE001, PERF203
            continue

    return out


# =================================================
# FETCH HISTORICAL STOCK DATA
# =================================================

def get_stock_data(
    ticker,
    period="10y",
    interval="1d",
    start=None,
    end=None
):

    try:

        print(
            f"\nDownloading data for {ticker}..."
        )


        # =================================================
        # DOWNLOAD USING DATE RANGE
        # =================================================

        if start is not None:

            df = yf.download(

                ticker,

                start=start,

                end=end,

                interval=interval,

                auto_adjust=True,

                progress=False,

                threads=False
            )


        # =================================================
        # DOWNLOAD USING PERIOD
        # =================================================

        else:

            df = yf.download(

                ticker,

                period=period,

                interval=interval,

                auto_adjust=True,

                progress=False,

                threads=False
            )


        # =================================================
        # CHECK IF DATA EXISTS
        # =================================================

        if df is None or df.empty:

            raise ValueError(
                f"No data found for {ticker}"
            )


        # =================================================
        # FIX MULTI-INDEX COLUMNS
        # =================================================

        if hasattr(
            df.columns,
            "levels"
        ):

            df.columns = df.columns.get_level_values(
                0
            )


        # =================================================
        # REMOVE EMPTY ROWS
        # =================================================

        df = df.dropna(
            how="all"
        )


        # =================================================
        # CHECK DATA AGAIN
        # =================================================

        if df.empty:

            raise ValueError(
                f"Downloaded data is empty for {ticker}"
            )


        # =================================================
        # SUCCESS MESSAGE
        # =================================================

        print(
            f"Successfully fetched "
            f"{len(df)} records."
        )


        print(
            f"Data range: "
            f"{df.index.min().date()} "
            f"to "
            f"{df.index.max().date()}"
        )


        return df


    except Exception as e:

        print(
            f"\nError fetching data: {e}"
        )

        return None