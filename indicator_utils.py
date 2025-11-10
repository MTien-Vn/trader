import numpy as np
import pandas as pd

def calculate_ema(prices, period):
    """Calculates Exponential Moving Average (EMA)."""
    if len(prices) < period:
        # Handle case where insufficient history is available for a true EMA
        return np.mean(prices) if len(prices) > 0 else 0
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values[-1]

def calculate_rsi(closes, period=14):
    """
    FIXED: Calculates Relative Strength Index (RSI) using robust Pandas EWMA 
    to prevent NaNs during the recursive forecasting loop.
    """
    closes_series = pd.Series(closes)
    
    if len(closes_series) <= period:
        return 50.0 
    
    diff = closes_series.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    # Use EWM for average gain/loss over the full series
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    # RS calculation
    # np.seterr(divide='ignore', invalid='ignore') # Suppress division by zero warning
    rs = avg_gain / avg_loss
    
    # Handle the case where avg_loss is 0 (RSI=100) or avg_gain is 0 (RSI=0)
    # Using np.inf for division by zero ensures 100
    rs.loc[avg_loss == 0] = np.inf
    # rs.loc[avg_gain == 0] = 0 # Not strictly needed as 0 / positive_number = 0
    
    rsi = 100 - (100 / (1 + rs))
    # np.seterr(divide='warn', invalid='warn') # Restore warning settings
    
    # Ensure the result is a number, defaulting to 50 if any NaN crept in (e.g., from initial periods)
    return rsi.values[-1] if not np.isnan(rsi.values[-1]) else 50.0

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD, MACD Signal, and MACD Hist. Returns the last value of each."""
    
    if len(closes) < slow_period:
        # Not enough history for the slow EMA (26). Return neutral values.
        return 0.0, 0.0, 0.0

    closes_series = pd.Series(closes)
    
    # 1. Calculate EMAs
    ema_fast = closes_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closes_series.ewm(span=slow_period, adjust=False).mean()
    
    # 2. Calculate MACD Line
    macd = ema_fast - ema_slow
    
    # 3. Calculate Signal Line (EMA of the MACD Line)
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # 4. Calculate MACD Histogram
    macd_hist = macd - macd_signal
    
    return macd.values[-1], macd_signal.values[-1], macd_hist.values[-1]

def calculate_atr(highs, lows, closes, period=14):
    """Calculates Average True Range (ATR)."""
    if len(closes) < 2:
        return 0.0

    # Ensure all inputs are NumPy arrays or lists of the same length
    h = np.array(highs[-len(closes):])
    l = np.array(lows[-len(closes):])
    c_prev = np.array(closes[-len(closes):])[:-1]
    c = np.array(closes[-len(closes):])[1:] # Current close is c[i], previous close is c_prev[i]

    # True Range (TR) calculation (starts from the second day)
    tr_h_l = h[1:] - l[1:]
    tr_h_pc = np.abs(h[1:] - c_prev)
    tr_l_pc = np.abs(l[1:] - c_prev)
    
    TR_values = np.maximum.reduce([tr_h_l, tr_h_pc, tr_l_pc])
    
    if len(TR_values) < period:
        # Use simple average if not enough history for EMA
        return np.mean(TR_values) if len(TR_values) > 0 else 0.0

    # Use EWMA for ATR calculation
    atr_series = pd.Series(TR_values).ewm(span=period, adjust=False).mean()
    return atr_series.values[-1]

def calculate_std_dev_return(closes, period=10):
    """Calculates Standard Deviation of Log Returns."""
    closes_series = pd.Series(closes)
    
    if len(closes_series) <= period:
        return 0.0
        
    # Calculate Log Returns
    returns = np.log(closes_series / closes_series.shift(1))
    
    # Calculate rolling standard deviation, then take the last value
    std_dev = returns.rolling(window=period).std()
    
    return std_dev.values[-1] if not np.isnan(std_dev.values[-1]) else 0.0



def recalculate_non_target_features_production(extended_data_history, next_predictions):
    """
    Recalculates all non-target technical indicators for the next day 
    using the predicted closing price.
    
    Args:
        sequence_original_data (np.array): The last TIME_STEP days of unscaled data.
        next_close (float): The predicted closing price for the next day.

    Returns:
        np.array: Unscaled values for the non-target features for the next day.
    """

    next_close = next_predictions['close']
    next_high = next_predictions['high']
    next_low = next_predictions['low']

    # Extend the close history
    closes_history = np.append(extended_data_history['close'].values, next_close)
    
    # Extend the high/low history (assuming High_T = Low_T = Close_T)
    highs_history = np.append(extended_data_history['high'].values, next_high)
    lows_history = np.append(extended_data_history['low'].values, next_low)
    
    # --- Calculation ---
    
    # 1. MA20 (Uses the last 20 periods)
    ma20_calc = np.mean(closes_history[-20:])
    
    # 2. RSI (Uses the full history for accurate EWMA initialization)
    rsi_calc = calculate_rsi(closes_history, period=14)
    
    # 3. MACD, MACD_Signal, MACD_Hist (Uses the full history)
    macd_calc, macd_signal_calc, macd_hist_calc = calculate_macd(closes_history)

    # 4. ATR (Requires high, low, close history)
    atr_calc = calculate_atr(highs_history, lows_history, closes_history, period=14)
    
    # 5. STD_DEV_10 (Requires close history)
    std_dev_calc = calculate_std_dev_return(closes_history, period=10)
    
    # Order the results according to NON_TARGET_FEATURES: 
    calculated_values = np.array([
        ma20_calc, 
        rsi_calc, 
        macd_calc, 
        macd_signal_calc, 
        macd_hist_calc,
        atr_calc,
        std_dev_calc
    ])

    return calculated_values.flatten()
