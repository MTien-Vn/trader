import requests
import pandas as pd
import os
import time
from calculate_technical_indicators import calculate_technical_indicators

def download_cafef_price_history(symbol, start_date_str, end_date_str, output_filename):
    """
    Downloads historical price data from the Cafef API for a given stock symbol
    and saves it to a CSV file, adapted for the nested JSON response structure.

    Args:
        symbol (str): The stock ticker (e.g., 'FPT').
        start_date_str (str): Start date in 'dd/mm/yyyy' format.
        end_date_str (str): End date in 'dd/mm/yyyy' format.
        output_filename (str): The name of the CSV file to save the data to.
    """
    # The base URL for the historical data API
    BASE_URL = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    
    # List to store all fetched data records
    all_records = []
    
    # Pagination setup
    page_index = 1
    page_size = 200 # Set a reasonably large page size for fewer requests
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


    # Save the final DataFrame to a CSV file
    df_final.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"\n✅ Successfully downloaded {len(df_final)} records for {symbol}.")
    print(f"Data saved to {output_filename}")




if __name__ == '__main__':
    # --- Configuration ---
    STOCK_SYMBOL = "VIX"
    # Use 'yyyy-mm-dd' format for dates
    START_DATE = "2021/01/01" 
    # END_DATE = datetime.now().strftime("%Y-%m-%d") 
    END_DATE = "2025-11-05" 
    filename = f"{STOCK_SYMBOL}_price_history.csv"
    # ---------------------
    # Ensure the folder exists
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)

    # Full path for the CSV file
    file_path = os.path.join(folder_path, filename)

    # Run the function
    download_cafef_price_history(STOCK_SYMBOL, START_DATE, END_DATE, file_path)
    calculate_technical_indicators(file_path)