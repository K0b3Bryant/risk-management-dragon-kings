import numpy as np
import pandas as pd

def rolling_volatility(prices, window=50):
    return prices.pct_change().rolling(window).std()

def detect_anomalies(prices, threshold=2.0):
    vol = rolling_volatility(prices)
    anomalies = (vol > vol.mean() + threshold * vol.std()).astype(int)
    return anomalies
