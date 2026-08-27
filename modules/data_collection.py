import yfinance as yf


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


# =================================================
# FETCH COMPANY INFORMATION
# =================================================

def get_stock_info(
    ticker
):

    try:

        print(
            f"\nFetching company information "
            f"for {ticker}..."
        )


        stock = yf.Ticker(
            ticker
        )


        info = stock.info


        return info


    except Exception as e:

        print(
            f"\nError fetching company information: {e}"
        )

        return None