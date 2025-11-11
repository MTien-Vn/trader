# config.py

import os

# --- Global Hyperparameters (Fixed) ---
TIME_STEP = 20 
TRAIN_SPLIT_PERCENT = 0.8
# NOTE: Ensure these features match your input data columns
FEATURES = ['VNINDEX_close', 'VNINDEX_open','VNINDEX_high', 'VNINDEX_low', 'VNINDEX_volume', 'close', 'volume', 'open', 'high', 'low', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'STD_DEV_10']
TARGET_COLUMNS = ['VNINDEX_close', 'VNINDEX_open','VNINDEX_high', 'close', 'VNINDEX_low', 'VNINDEX_volume', 'volume', 'high', 'low', 'open'] 
NON_TARGET_FEATURES = [f for f in FEATURES if f not in TARGET_COLUMNS] 
VALIDATION_SPLIT = 0.1
ACTIVATION = 'linear'
OPTIMIZER = 'adam'
EPOCHS = 50
BATCH_SIZE = 32
FORECAST_DAYS = 7 
# Max history needed for MACD Slow EMA (26 periods)
MAX_INDICATOR_LOOKBACK = 26 

# --- KerasTuner Hyperparameter Search Settings ---
MAX_TRIALS = 30
EXECUTIONS_PER_TRIAL = 1 
OBJECTIVE_METRIC = 'val_loss' 
PROJECT_NAME = 'lstm_stock_bo'
TUNER_DIR = 'tuning_results'

# LSTM/Dense Layer Search Space
MAX_LAYER = 5
MAX_NEURONS_LAYER = 512
MIN_NEURONS_LAYER = 32
STEP_NEURONS_LAYER = 32

# ConvLSTM Specifics
CONV_FILTER_POOL = [32, 64] # Filters for the ConvLSTM layer
CONV_KERNEL_SIZE = (1, 3)    # Kernel size (time_steps, features)

# Callbacks
PATIENCE_EARLY_STOPPING = 10
PATIENCE_LR_SCHEDULING = 5

# --- Data File Path ---
DATA_FILE_PATH = 'data/PLX_price_history.csv'

# Ensure the 'data' directory exists for local testing
if not os.path.exists('data'):
    os.makedirs('data')

STOCK_SYMBOLS = ['PLX','PC1']