import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
from functools import lru_cache

@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
    """
    Fetch historical data for a given ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return {"error": "No data found for ticker"}
        
        # Reset index to make Date a column
        hist.reset_index(inplace=True)
        
        # Normalize Date/Datetime column
        if 'Datetime' in hist.columns:
            hist.rename(columns={'Datetime': 'Date'}, inplace=True)
            # Convert to string for JSON serialization
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        elif 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        
        # Convert to list of dicts for JSON response
        data = hist.to_dict(orient="records")
        
        # Get info
        info = stock.info
        
        return {
            "ticker": ticker,
            "info": {
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "currency": info.get("currency"),
                "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
                "logo_url": info.get("logo_url") or "https://logo.clearbit.com/" + (info.get("website") or "google.com").replace("https://", "").replace("http://", "").split("/")[0],
                "marketCap": info.get("marketCap"),
                "peRatio": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "beta": info.get("beta"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "dividendYield": info.get("dividendYield"),
                "averageVolume": info.get("averageVolume"),
                "profitMargins": info.get("profitMargins"),
            },
            "history": data
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {"error": str(e)}

import requests

def search_symbol(query: str):
    """
    Search for a symbol using Yahoo Finance API.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if "quotes" in data:
            results = []
            for quote in data["quotes"]:
                # Filter out irrelevant types if needed, or keep all
                results.append({
                    "symbol": quote.get("symbol"),
                    "name": quote.get("longname") or quote.get("shortname") or quote.get("symbol"),
                    "exchange": quote.get("exchange"),
                    "type": quote.get("quoteType")
                })
            return results
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return [{"symbol": query.upper(), "name": query.upper(), "exchange": "Unknown", "type": "Unknown"}]
