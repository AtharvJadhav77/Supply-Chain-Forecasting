# news/news_fetcher.py

import requests
from datetime import datetime, timedelta

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_news(commodity, days=7):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    params = {
        "q": commodity,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    articles = data.get("articles", [])
    headlines = [article["title"] for article in articles if article["title"]]

    return headlines
