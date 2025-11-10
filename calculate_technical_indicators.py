import pandas as pd

import pandas as pd

def load_and_prepare_data(csv_filename):
    """
    Loads the CSV file, parses the date, sorts the data, and returns the DataFrame.
    
    Args:
        csv_filename (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The prepared DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(csv_filename)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df)} records from {csv_filename}")
        
        # Sort by date in ascending order, crucial for time series calculations
        df = df.sort_values(by='date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{csv_filename}'.")
        return None
    except Exception as e:
        print(f"❌ An error occurred during data loading: {e}")
        return None

def calculate_momentum_indicators(df):
    """
    Calculates MA20, MACD, and RSI.
    
    Args:
        df (pd.DataFrame): The input DataFrame with 'close' prices.
        
    Returns:
        pd.DataFrame: The DataFrame with new indicator columns.
    """
    # --- 1. Calculate MA20 (20-day Simple Moving Average) ---
    df['MA20'] = df['close'].rolling(window=20).mean()

    # --- 2. Calculate MACD (Moving Average Convergence Divergence) ---
    EMA_12 = df['close'].ewm(span=12, adjust=False).mean()
    EMA_26 = df['close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = EMA_12 - EMA_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # --- 3. Calculate RSI (Relative Strength Index) (14-day) ---
    period = 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use EWMA for smoothing (standard RSI calculation)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    RS = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    return df

def calculate_volatility_indicators(df):
    """
    Calculates ATR and Standard Deviation of Returns (STD_DEV_10).
    
    Args:
        df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' prices.
        
    Returns:
        pd.DataFrame: The DataFrame with new indicator columns.
    """
    # --- 1. Calculate ATR (Average True Range) (14-day) ---
    # True Range (TR) calculation
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    df['TR'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # ATR as an EWMA of TR
    atr_period = 14
    df['ATR'] = df['TR'].ewm(com=atr_period - 1, adjust=False).mean()
    df = df.drop('TR', axis=1)

    # --- 2. Calculate Standard Deviation of Returns (STD_DEV_10) ---
    std_period = 10
    # Calculate daily percentage returns
    df['Daily_Return'] = df['close'].pct_change() * 100 
    
    # Calculate the rolling standard deviation of the daily returns
    df['STD_DEV_10'] = df['Daily_Return'].rolling(window=std_period).std()
    df = df.drop('Daily_Return', axis=1)
    
    return df

def calculate_technical_indicators(df, file_path):
    """
    Orchestrates the loading, calculation, formatting, and saving of 
    technical indicators.
    
    Args:
        csv_filename (str): The path to the CSV file containing price history.
    """
    if df is None:
        return # Exit if data loading failed

    # Calculate indicators
    df = calculate_momentum_indicators(df)
    df = calculate_volatility_indicators(df)

    # Rounding for presentation
    df = df.round({
        'MA20': 2, 'RSI': 2, 'MACD': 3, 'MACD_Signal': 3, 'MACD_Hist': 3,
        'ATR': 3, 'STD_DEV_10': 3 
    })

    # --- Output and Save ---
    print("\nDataFrame Info:")
    df.info()
    print("\nFirst 10 rows (most recent dates) with calculated indicators:")
    print(df.head(10).to_markdown(index=False))

    try:
        df.to_csv(file_path, index=False)
        print(f"\n✅ Data successfully saved to '{file_path}'")
    except Exception as e:
        print(f"❌ An error occurred during file saving: {e}")

    return df

# Example Usage (replace 'data.csv' with your actual file name):
# calculate_technical_indicators('/home/vtp-tiennm/Documents/learn/trading/trader/data/PLX_price_history.csv')