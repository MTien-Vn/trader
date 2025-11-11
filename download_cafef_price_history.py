from datetime import datetime
import requests
import pandas as pd
import numpy as np
import os
import time
from calculate_technical_indicators import calculate_technical_indicators
from config import DATA_FILE_PATH, FEATURES, FOLDER_PATH, STOCK_SYMBOLS

def download_cafef_price_history(symbol, start_date_str, end_date_str):
    """
    Downloads historical price data from the Cafef API for a given stock symbol
    and saves it to a CSV file, adapted for the nested JSON response structure.

    Args:
        symbol (str): The stock ticker (e.g., 'FPT').
        start_date_str (str): Start date in 'dd/mm/yyyy' format.
        end_date_str (str): End date in 'dd/mm/yyyy' format.
    """
    # The base URL for the historical data API
    BASE_URL = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    
    # List to store all fetched data records
    all_records = []
    
    # Pagination setup
    page_index = 1
    page_size = 400 # Set a reasonably large page size for fewer requests
    total_pages = 1 # Start with 1, will be updated by the first response

    print(f"Starting download for symbol: {symbol}...")

    # --- KEY MAPPING ---
    # Map the Vietnamese JSON keys to the required English column names.
    COLUMN_MAPPING = {
        'Ngay': 'date',                 # Date
        'GiaDongCua': 'close',          # Closing Price
        'GiaMoCua': 'open',             # Opening Price
        'GiaCaoNhat': 'high',           # Highest Price
        'GiaThapNhat': 'low',           # Lowest Price
        'KhoiLuongKhopLenh': 'volume',  # Trading Volume
    }
    # -------------------

    # Loop through all pages until the current page index exceeds the total pages
    while page_index <= total_pages:
        # Define the query parameters for the API call
        params = {
            "Symbol": symbol,
            "StartDate": start_date_str,
            "EndDate": end_date_str,
            "PageIndex": page_index,
            "PageSize": page_size
        }

        try:
            # Make the GET request to the API
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            
            # --- CRITICAL ADAPTATION: Handle Nested Structure ---
            api_data_payload = raw_data.get('Data', {})
            
            # The list of price records is now nested under api_data_payload['Data']
            records = api_data_payload.get('Data', [])
            
            # TotalPage count is usually calculated based on TotalCount / PageSize, 
            # but sometimes the API provides a 'TotalPage' key. 
            # We'll use a common pattern where the total count is provided.
            total_count = api_data_payload.get('TotalCount', 0)
            
            # Calculate total pages based on TotalCount
            if total_count > 0:
                total_pages = (total_count + page_size - 1) // page_size
            # ----------------------------------------------------

            if not records:
                if page_index == 1:
                    print(f"Warning: No data found for {symbol} in the specified date range.")
                break

            all_records.extend(records)
            print(f"Page {page_index} of {total_pages} downloaded. Total records so far: {len(all_records)}")
            
            page_index += 1
            
            # Pause briefly to be polite to the server
            time.sleep(0.5) 

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request for page {page_index}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    if not all_records:
        print(f"Failed to retrieve any data for {symbol}. Exiting.")
        return

    # --- Data Processing ---
    df = pd.DataFrame(all_records)
    
    required_keys = list(COLUMN_MAPPING.keys())
    
    # Select and rename the required columns
    df_final = df[required_keys].copy()
    df_final = df_final.rename(columns=COLUMN_MAPPING)
    
    # Insert the 'code' column as the first column
    df_final.insert(0, 'code', symbol)
    
    # 1. Date formatting (from 'dd/mm/yyyy' to 'YYYY-MM-DD')
    df_final['date'] = pd.to_datetime(df_final['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    
    # 2. Convert price and volume columns to numeric
    numeric_cols = ['close', 'open', 'high', 'low', 'volume']
    
    for col in numeric_cols:
        # Step 1: Convert to string to handle mixed types
        # Step 2: Remove all characters that are NOT digits or a decimal point. 
        # This handles potential string formatting from the API (like '59.5' or '1,046,700').
        df_final[col] = df_final[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        
        # Step 3: Convert to numeric, coercing errors to NaN
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    print(f"\n✅ Successfully downloaded {len(df_final)} records for {symbol}.")

    # Sort by date in ascending order, crucial for time series calculations
    df_final = df_final.sort_values(by='date').reset_index(drop=True)

    return df_final




if __name__ == '__main__':
    # Use 'yyyy-mm-dd' format for dates
    START_DATE = "2017-01-01" 
    END_DATE = datetime.now().strftime("%Y-%m-%d") 
    # END_DATE = "2025-11-06" 
    # ---------------------
    # Ensure the folder exists
    os.makedirs(FOLDER_PATH, exist_ok=True)

    all_data = []

    for stock_symbol in STOCK_SYMBOLS:
        filename = f"{stock_symbol}_price_history.csv"
        output_file = os.path.join(FOLDER_PATH, filename)

        current_df = download_cafef_price_history(stock_symbol, START_DATE, END_DATE)

        # current_df = pd.read_csv(output_file, parse_dates=True)
        # current_df['date'] = pd.to_datetime(current_df['date'])
        # current_df = current_df.sort_values(by='date').reset_index(drop=True)

        current_df = calculate_technical_indicators(current_df)

        # 🧹 Clean data
        current_df.dropna(inplace=True)           # remove rows with any NaN
        current_df.drop_duplicates(inplace=True)  # remove duplicate rows

        # 📊 Round all numeric columns to 2 decimal places
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        current_df[numeric_cols] = current_df[numeric_cols].round(2)

        # 🚫 Remove rows with zero values in any numeric column
        current_df = current_df[(current_df[numeric_cols] != 0).all(axis=1)]

        mask_invalid = np.isinf(current_df[numeric_cols]) | (current_df[numeric_cols].abs() > np.finfo(np.float64).max)
        current_df.drop(index=mask_invalid[mask_invalid.any(axis=1)].index, inplace=True)

        current_df['date'] = pd.to_datetime(current_df['date'])

        current_df.sort_values(by=['date'], inplace=True)

        all_data.append(current_df)

        current_df.to_csv(output_file, index=False)
        print(f"✅ Saved combined indicators to: {output_file}")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.drop_duplicates(inplace=True)  # remove duplicate rows

    final_df.to_csv(DATA_FILE_PATH, index=False)

    print(f"✅ Total rows saved: {len(final_df)}")




