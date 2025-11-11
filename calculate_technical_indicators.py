import pandas as pd

import numpy as np

# ======================
# --- RSI (Relative Strength Index)
# ======================
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


# ======================
# --- MACD (Moving Average Convergence Divergence)
# ======================
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, min_periods=signal).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


# ======================
# --- ATR (Average True Range)
# ======================
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    return df


# ======================
# --- STD_DEV (Rolling Standard Deviation)
# ======================
def calculate_stddev(df, period=10):
    df['STD_DEV'] = df['close'].rolling(window=period).std()
    return df


# ======================
# --- MFI (Money Flow Index)
# ======================
def calculate_mfi(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)

    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    df['MFI'] = mfi
    return df


# ======================
# --- VROC (Volume Rate of Change)
# ======================
def calculate_vroc(df, period=14):
    df['VROC'] = ((df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)) * 100
    return df


# ======================
# --- CMF (Chaikin Money Flow)
# ======================
def calculate_cmf(df, period=20):
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    df['CMF'] = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return df

def calculate_percent_change(df):
    """
    Adds a column 'PCT_CHANGE' that represents the percent change in closing price
    compared to the previous day.
    """
    df = df.copy()
    df['PCT_CHANGE'] = df['close'].pct_change() * 100
    return df

def calculate_technical_indicators(df):
    df = df.copy()
    df = calculate_percent_change(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_atr(df)
    df = calculate_stddev(df)
    df = calculate_mfi(df)
    df = calculate_vroc(df)
    df = calculate_cmf(df)
    return df

# Example Usage (replace 'data.csv' with your actual file name):
# calculate_technical_indicators('/home/vtp-tiennm/Documents/learn/trading/trader/data/PLX_price_history.csv')