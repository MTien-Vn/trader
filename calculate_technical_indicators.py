import pandas as pd

def calculate_technical_indicators(csv_filename):
    """
    Loads price history data and calculates RSI, MACD, and MA(20).
    
    Args:
        csv_filename (str): The path to the CSV file containing price history.
    """
    try:
        # 1. Load the data
        # Ensure 'date' column is parsed as datetime and set as index for time series analysis
        df = pd.read_csv(csv_filename)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df)} records from {csv_filename}")
        
        df = df.sort_values(by='date').reset_index(drop=True)

        # --- 1. Calculate MA20 (20-day Simple Moving Average) ---
        df['MA20'] = df['close'].rolling(window=20).mean()

        # --- 2. Calculate MACD (Moving Average Convergence Divergence) ---
        # Standard periods: Fast EMA (12), Slow EMA (26), Signal Line (9).
        EMA_12 = df['close'].ewm(span=12, adjust=False).mean()
        EMA_26 = df['close'].ewm(span=26, adjust=False).mean()

        df['MACD'] = EMA_12 - EMA_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # --- 3. Calculate RSI (Relative Strength Index) ---
        # Standard period for RSI is 14 days.

        # Calculate price change
        delta = df['close'].diff()

        # Get positive and negative changes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate EWMA of gains and losses (alpha = 1/period)
        # For standard RSI, the first 14 values are calculated as a simple average,
        # and subsequent values use the EWMA formula. pandas .ewm(com=period-1)
        # provides the standard way to calculate the initial values as a simple average.
        period = 14
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        # Calculate Relative Strength (RS)
        RS = avg_gain / avg_loss

        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + RS))

        # Sort back by date in descending order for presentation
        df = df.sort_values(by='date', ascending=False).reset_index(drop=True)

        # Select and format the relevant columns for output
        output_df = df[['date','volume', 'close', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']]
        output_df = output_df.round({'MA20': 2, 'RSI': 2, 'MACD': 3, 'MACD_Signal': 3, 'MACD_Hist': 3})

        # Display info and head
        print("DataFrame Info:")
        output_df.info()
        print("\nFirst 10 rows (most recent dates) with calculated indicators:")
        print(output_df.head(10).to_markdown(index=False))

        # Save the resulting DataFrame to a new CSV file
        output_filename = csv_filename.replace(".csv", "_with_indicators.csv")
        output_df.to_csv(output_filename, index=False)

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{csv_filename}'. Please ensure your download script ran successfully.")
    except Exception as e:
        print(f"❌ An error occurred during indicator calculation: {e}")
