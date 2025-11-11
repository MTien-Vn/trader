# data_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import * # Import all global variables

def load_and_clean_data(file_path=DATA_FILE_PATH):
    """Loads and performs basic cleaning/validation on the stock data."""
    df = pd.read_csv(file_path, parse_dates=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")
    
    df.dropna(subset=TARGET_COLUMNS, inplace=True)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].mean())
    df.dropna(subset=FEATURES, inplace=True)

    df = df.drop(columns=['date'])
    print(f"✅ Data Loaded and Cleaned. Total rows: {len(df)}. Using features: {FEATURES}")
    return df

def create_sequences(data, time_step, features, target_columns):
    """
    Creates the time-stepped sequences (X) and next-day targets (Y) 
    for the LSTM model.
    """
    X, Y = [], []
    target_indices = [features.index(col) for col in target_columns]
    
    # We stop earlier than the original code to ensure the target is not out of bounds
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step, target_indices]) 
        
    return np.array(X), np.array(Y)

def preprocess_data(file_path, time_step, features, target_columns):
    """
    Handles time-aware splitting, scaling (NO DATA LEAKAGE), and sequence creation.
    
    Returns: X_train, X_test, y_train, y_test, full_scaler, target_scaler, final_df_features
    """
    df = load_and_clean_data(file_path)
    df_features = df[features]
    full_data = df_features.values
    
    # 1. Perform initial split based on features index (before sequence creation)
    train_size_rows = int(len(full_data) * TRAIN_SPLIT_PERCENT)
    
    train_data = full_data[:train_size_rows]
    test_data = full_data[train_size_rows:]
    
    # 2. Initialize and FIT scalers ONLY on TRAINING DATA (Leakage Prevention)
    full_data_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = full_data_scaler.fit_transform(train_data)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit target scaler on the training part of the target columns
    target_cols_indices = [features.index(col) for col in target_columns]
    target_scaler.fit_transform(train_data[:, target_cols_indices])
    
    # 3. Transform the full dataset for sequence creation
    scaled_full_data = full_data_scaler.transform(full_data)

    # 4. Create sequences from the scaled full data
    X, y = create_sequences(scaled_full_data, time_step, features, target_columns)
    
    # 5. Final split of sequences (time-aware)
    train_size_seq = int(len(X) * TRAIN_SPLIT_PERCENT)
    X_train, X_test = X[:train_size_seq], X[train_size_seq:]
    y_train, y_test = y[:train_size_seq], y[train_size_seq:]
    
    print(f"✅ Data Split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    return X_train, X_test, y_train, y_test, full_data_scaler, target_scaler, df_features

def get_target_indices(features=FEATURES, target_columns=TARGET_COLUMNS):
    """Helper to get indices of target columns within the full feature set."""
    return [features.index(col) for col in target_columns]