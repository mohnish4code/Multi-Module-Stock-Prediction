import feedparser
from urllib.parse import quote


# =================================================
# COMPANY SEARCH QUERIES
# =================================================

COMPANY_NEWS_QUERIES = {

    "TCS": (
        "Tata Consultancy Services "
        "TCS India stock"
    ),

    "RELIANCE": (
        "Reliance Industries "
        "India stock"
    ),

    "INFY": (
        "Infosys India stock"
    ),

    "HDFCBANK": (
        "HDFC Bank India stock"
    ),

    "ICICIBANK": (
        "ICICI Bank India stock"
    ),

    "SBIN": (
        "State Bank of India SBI "
        "stock"
    ),

    "WIPRO": (
        "Wipro India stock"
    ),

    "HINDUNILVR": (
        "Hindustan Unilever "
        "HUL India stock"
    ),

    "ITC": (
        "ITC Limited India stock"
    ),

    "BHARTIARTL": (
        "Bharti Airtel India stock"
    )

}


# =================================================
# GET COMPANY NEWS
# =================================================

def get_stock_news(
    company_name,
    max_news=10
):
    """
    Fetches recent company-specific news headlines
    from Google News RSS.
    """

    print(
        "Fetching recent company news..."
    )


    # -------------------------------------------------
    # STANDARDIZE COMPANY NAME
    # -------------------------------------------------

    company_key = company_name.upper()


    # -------------------------------------------------
    # GET SEARCH QUERY
    # -------------------------------------------------

    search_query = COMPANY_NEWS_QUERIES.get(

        company_key,

        f"{company_name} India stock"

    )


    # -------------------------------------------------
    # ENCODE QUERY
    # -------------------------------------------------

    query = quote(
        search_query
    )


    # -------------------------------------------------
    # GOOGLE NEWS RSS URL
    # -------------------------------------------------

    rss_url = (

        "https://news.google.com/rss/search?"

        f"q={query}"

        "&hl=en-IN"

        "&gl=IN"

        "&ceid=IN:en"

    )


    # -------------------------------------------------
    # FETCH RSS FEED
    # -------------------------------------------------

    try:

        feed = feedparser.parse(
            rss_url
        )


        news = []


        # -------------------------------------------------
        # EXTRACT NEWS
        # -------------------------------------------------

        for entry in feed.entries[:max_news]:

            title = entry.get(
                "title",
                ""
            )


            link = entry.get(
                "link",
                ""
            )


            source = entry.get(
                "source",
                {}
            )


            if isinstance(source, dict):

                publisher = source.get(
                    "title",
                    "Unknown"
                )

            else:

                publisher = "Unknown"


            if title:

                news.append(

                    {

                        "title": title,

                        "publisher": publisher,

                        "link": link

                    }

                )


        return news


    except Exception as error:

        print(
            f"Error fetching news: {error}"
        )

        return []