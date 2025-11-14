# config.py

import os

# --- Global Hyperparameters (Fixed) ---
TIME_STEP = 20 
TRAIN_SPLIT_PERCENT = 0.8
# NOTE: Ensure these features match your input data columns
FEATURES = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'STD_DEV', 'MFI', 'VROC', 'CMF', 'PCT_CHANGE']
TARGET_COLUMNS =  ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'STD_DEV', 'MFI', 'VROC', 'CMF', 'PCT_CHANGE']
NON_TARGET_FEATURES = [f for f in FEATURES if f not in TARGET_COLUMNS] 
VALIDATION_SPLIT = 0.1
ACTIVATION = 'linear'
OPTIMIZER = 'adam'
EPOCHS = 50
BATCH_SIZE = 128
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


# Use 'yyyy-mm-dd' format for dates
START_DATE = "2025-08-01"
END_DATE = "2025-11-11" 

FOLDER_DATA_PATH = 'data_training'
# FOLDER_DATA_PATH = 'data_predict'

FOLDER_PREDICT_PATH = 'data_predict'
FOLDER_RESULT_PATH = 'summary'

# --- Data File Path ---
DATA_TRAINING_FILE_PATH = f'{FOLDER_DATA_PATH}/price_history.csv'

STOCK_SYMBOLS = ['HPG', 'VIC', 'VHM', 'SSI', 'VIX', 'VND', 'SHB', 'MSN', 'GAS', 'POW',
                 'VNM', 'BID', 'VRE', 'VCB', 'SBT', 'HAG', 'VCI', 'SAB', 'GVR', 'NAB',
                 'EIB', 'VPB', 'PDR', 'GEX', 'KBC', 'TCH', 'BCM', 'DPM', 'DSE', 'DIG', 
                 'VCG', 'DXG', 'HSG', 'FTS', 'KDH', 'STB', 'PVD', 'CTG', 'DCM', 'CII', 
                 'MBB', 'VGC', 'NKG', 'HHV', 'PVT', 'FPT', 'VHC', 'GEE', 'DXS', 'DBC', 
                 'HT1', 'VSC', 'BVH', 'BSI', 'TPB', 'HCM', 'SJS', 'PPC', 'DGC', 'SSB', 
                 'VJC', 'PC1', 'VPI', 'ANV', 'LPB', 'EVF', 'SIP', 'HDG', 'KOS', 'NT2', 
                 'CTS', 'KDC', 'BWE', 'HDC', 'TLG', 'OCB', 'DGW', 'PLX', 'PAN', 'VTP', 
                 'CTR', 'MWG', 'SZC', 'ACB', 'FRT', 'CMG', 'TCB', 'SCS', 'NLG', 'IMP' ]

PREDICT_STOCK_SYMBOLS = ['PLX','PC1']

# --- CHOOSE MODEL HERE: 'LSTM', 'Bi-LSTM', 'ConvLSTM', 'GRU', 'attention' ---
SELECTED_MODEL_TYPE = 'LSTM'

BEST_MODEL = 'best_model_lstm'

FULL_SCALER_PATH = os.path.join(BEST_MODEL, f'full_scaler_{SELECTED_MODEL_TYPE}.joblib')
TARGET_SCALER_PATH = os.path.join(BEST_MODEL, f'target_scaler_{SELECTED_MODEL_TYPE}.joblib')
MODEL_SAVE_PATH = os.path.join(BEST_MODEL, f'{PROJECT_NAME}_{SELECTED_MODEL_TYPE}.keras')

