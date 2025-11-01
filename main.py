import yfinance as yf
import pandas as pd

def load_prices(ticker="SPY", start="2010-01-01", end=None, auto_adjust=True):
    """
    Henter justerede lukkepriser (dividender/split taget højde for hvis auto_adjust=True).
    Returnerer en Series navngivet 'Close' så resten af koden virker uændret.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"Ingen data for {ticker}.")
    px = df["Close"].astype(float).rename("Close")
    return px

if __name__ == "__main__":
    # Kort test
    s = load_prices("SPY", start="2020-01-01")
    print(s.head())