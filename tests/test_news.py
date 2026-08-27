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

from modules.news_collection import get_stock_news


print("\nFetching latest news...\n")


headlines = get_stock_news(

    company_name="Tata Consultancy Services",

    max_news=10
)


print("\n" + "=" * 60)

print("LATEST STOCK NEWS")

print("=" * 60)


if not headlines:

    print(
        "No recent news found."
    )

else:

    for i, headline in enumerate(
        headlines,
        start=1
    ):

        print(
            f"{i}. {headline}"
        )