import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import kerastuner as kt
import os 
import shutil
import copy 

# --- Global Hyperparameters (Fixed) ---
TIME_STEP = 20 
TRAIN_SPLIT_PERCENT = 0.8
FEATURES = ['volume', 'close', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
TARGET_COLUMNS = ['close', 'volume'] 
NON_TARGET_FEATURES = [f for f in FEATURES if f not in TARGET_COLUMNS] 
VALIDATION_SPLIT = 0.1
ACTIVATION = 'linear'
OPTIMIZER = 'adam'
MAX_LAYER = 3
MAX_NEURONS_LAYER = 256
MIN_NEURONS_LAYER = 32
STEP_NEURONS_LAYER = 32
EPOCHS = 50
BATCH_SIZE = 32
FORECAST_DAYS = 7 

# --- KerasTuner Hyperparameter Search Settings (Unchanged) ---
MAX_TRIALS = 15     
EXECUTIONS_PER_TRIAL = 1 
OBJECTIVE_METRIC = 'val_loss' 
PROJECT_NAME = 'lstm_stock_bo'
TUNER_DIR = 'tuning_results'

# --- Data Loading and Preprocessing Functions (Unchanged) ---

def load_and_clean_data(file_path='stock_data.csv'):
    """Loads and performs basic cleaning/validation on the stock data."""
    df = pd.read_csv(file_path, parse_dates=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")
    
    df.dropna(subset=TARGET_COLUMNS, inplace=True)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].mean())
    
    if df[FEATURES].isnull().values.any():
         print("⚠️ WARNING: NaNs still present after imputation. Dropping remaining rows with NaNs.")
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
    
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step, target_indices]) 
        
    return np.array(X), np.array(Y)

def preprocess_data(df, time_step, features, target_columns):
    """
    Handles scaling and sequence creation.
    """
    full_data_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = full_data_scaler.fit_transform(df[features].values)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit_transform(df[target_columns].values)
    
    X, y = create_sequences(scaled_data, time_step, features, target_columns)
    
    return X, y, full_data_scaler, target_scaler, df[features]

def split_data(X, y, train_split_percent):
    """Performs a time-aware train/test split."""
    train_size = int(len(X) * train_split_percent)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"✅ Data Split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    return X_train, X_test, y_train, y_test

# --- Hypermodel Function (Unchanged) ---
def build_hypermodel(hp):
    """
    Defines the LSTM model structure and the hyperparameter search space.
    """
    model = Sequential()
    hp_layers = hp.Int('num_layers', min_value=1, max_value=MAX_LAYER, step=1)
    hp_units = hp.Int('units', min_value=MIN_NEURONS_LAYER, max_value=MAX_NEURONS_LAYER, step=STEP_NEURONS_LAYER)
    hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    for i in range(hp_layers):
        return_seq = (i < hp_layers - 1)
        if i == 0:
            model.add(LSTM(units=hp_units, return_sequences=return_seq, 
                           input_shape=(TIME_STEP, X_train.shape[2])))
        else:
            model.add(LSTM(units=hp_units, return_sequences=return_seq))
            
        model.add(Dropout(hp_dropout))
        
    n_targets = y_train.shape[1]
    model.add(Dense(units=n_targets, activation=ACTIVATION)) 
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error',
        metrics=['mse'] 
    )
    
    return model

def get_target_indices(features, target_columns):
    """Helper to get indices of target columns within the full feature set."""
    return [features.index(col) for col in target_columns]

# ====================================================================
# --- Production-Ready Indicator Calculation Functions ---
# ====================================================================

def calculate_ema(prices, period):
    """Calculates Exponential Moving Average (EMA)."""
    # prices is a NumPy array or Series
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values[-1]

def calculate_rsi(closes, period=14):
    """Calculates Relative Strength Index (RSI)."""
    # closes should be a NumPy array or Series
    
    diff = pd.Series(closes).diff()
    up = diff.clip(lower=0)
    down = -1 * diff.clip(upper=0)
    
    # Calculate initial EWMA (SMA for the first period)
    # The first 'period' values are calculated using SMA, then EWMA is applied.
    # We use a Series to leverage Pandas' efficient EWMA calculation.
    roll_up = up.rolling(period).mean().iloc[period-1]
    roll_down = down.rolling(period).mean().iloc[period-1]
    
    # Apply standard EWMA for subsequent steps
    for i in range(period, len(closes)):
        roll_up = (roll_up * (period - 1) + up.iloc[i]) / period
        roll_down = (roll_down * (period - 1) + down.iloc[i]) / period

    if roll_down == 0:
        rsi = 100.0 # Perfectly bullish
    else:
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
    return rsi

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD, MACD Signal, and MACD Hist. Returns the last value of each."""
    
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


def recalculate_non_target_features_production(sequence_original_data, next_close):
    """
    Recalculates all non-target technical indicators for the next day 
    using the predicted closing price.
    
    Args:
        sequence_original_data (np.array): The last TIME_STEP days of unscaled data.
        next_close (float): The predicted closing price for the next day.

    Returns:
        np.array: Unscaled values for the non-target features for the next day.
    """
    # Extract the 'close' price history
    close_index = FEATURES.index('close')
    
    # Sequence of closes (TIME_STEP values) + new predicted close (1 value)
    closes_history = np.append(sequence_original_data[:, close_index], next_close)
    
    # --- Calculation ---
    
    # 1. MA20 (Requires 20 periods)
    # We need the last 20 closing prices, which is possible since TIME_STEP=20.
    ma20_calc = np.mean(closes_history[-20:])
    
    # 2. RSI (Requires 14 periods)
    rsi_calc = calculate_rsi(closes_history, period=14)
    
    # 3. MACD, MACD_Signal, MACD_Hist
    macd_calc, macd_signal_calc, macd_hist_calc = calculate_macd(closes_history)
    
    # Order the results according to NON_TARGET_FEATURES: 
    # ['MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
    calculated_values = np.array([
        ma20_calc, 
        rsi_calc, 
        macd_calc, 
        macd_signal_calc, 
        macd_hist_calc
    ])
    
    return calculated_values.flatten()

# ====================================================================
# --- Forecasting Function (Updated) ---
# ====================================================================

def predict_next_n_days(model, initial_scaled_sequence, initial_original_sequence, full_data_scaler, target_scaler, n_days, features, target_columns):
    """
    Performs recursive multi-step forecasting for n_days using 
    recalculated indicators.
    """
    current_scaled_sequence = copy.deepcopy(initial_scaled_sequence)
    current_original_sequence = copy.deepcopy(initial_original_sequence)
    forecasted_predictions = []
    
    target_indices = get_target_indices(features, target_columns)
    non_target_indices = [features.index(col) for col in NON_TARGET_FEATURES]

    for day in range(n_days):
        
        # 1. Predict the next step (Close, Volume) - result is scaled
        X_next = current_scaled_sequence.reshape(1, TIME_STEP, len(features))
        scaled_prediction = model.predict(X_next, verbose=0)
        
        # 2. Inverse transform the prediction to get unscaled Close/Volume
        final_prediction_unscaled = target_scaler.inverse_transform(scaled_prediction)[0]
        next_close_unscaled = final_prediction_unscaled[target_columns.index('close')]
        next_volume_unscaled = final_prediction_unscaled[target_columns.index('volume')]

        # 3. Recalculate non-target features for the new step (unscaled)
        # We only need the predicted Close for the indicators
        next_non_target_unscaled = recalculate_non_target_features_production(
            current_original_sequence, next_close_unscaled
        )

        # 4. Construct the complete new point (unscaled)
        new_unscaled_data_point = np.zeros(len(features))
        new_unscaled_data_point[target_indices] = final_prediction_unscaled
        new_unscaled_data_point[non_target_indices] = next_non_target_unscaled
        
        # 5. Scale the new point to be used as input for the next step
        new_scaled_data_point = full_data_scaler.transform(new_unscaled_data_point.reshape(1, -1))[0]

        # 6. Update the sliding window sequences
        current_scaled_sequence = np.vstack([current_scaled_sequence[1:], new_scaled_data_point])
        current_original_sequence = np.vstack([current_original_sequence[1:], new_unscaled_data_point])
        
        # 7. Store the unscaled prediction for reporting
        forecasted_predictions.append(final_prediction_unscaled)
        
    return np.array(forecasted_predictions)


def evaluate_predict_and_forecast(model, X_test, y_test, target_scaler, last_data_df, full_data_scaler):
    """
    Evaluates the model on the test set, makes a final single-day prediction, 
    and performs a multi-day forecast.
    """
    
    # --- A. Test Evaluation (Single-Step Predictions) ---
    scaled_predictions = model.predict(X_test, verbose=0)
    predictions = target_scaler.inverse_transform(scaled_predictions)
    y_test_original = target_scaler.inverse_transform(y_test)
    actual_close = y_test_original[:, 0]
    predicted_close = predictions[:, 0]
    rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
    print(f"\n--- Evaluation Results (Test Set) ---")
    print(f"Root Mean Squared Error (RMSE) for Close Price: ${rmse:,.2f}")

    # --- B. Multi-Day Forecast ---
    
    last_sequence_original = last_data_df.tail(TIME_STEP).values
    scaled_last_sequence = full_data_scaler.transform(last_sequence_original)
    
    forecasted_results = predict_next_n_days(
        model, 
        scaled_last_sequence,          # Scaled sequence for LSTM input
        last_sequence_original,        # Original sequence for indicator recalculation
        full_data_scaler, 
        target_scaler, 
        n_days=FORECAST_DAYS, 
        features=FEATURES,
        target_columns=TARGET_COLUMNS
    )

    print(f"\n--- {FORECAST_DAYS}-Day Multi-Step Forecast (Recalculated Indicators) ---")
    
    forecast_df = pd.DataFrame(forecasted_results, columns=TARGET_COLUMNS)
    forecast_df.index = [f"Day +{i+1}" for i in range(FORECAST_DAYS)]
    
    forecast_df['close'] = forecast_df['close'].apply(lambda x: f"{x:,.2f} VND")
    forecast_df['volume'] = forecast_df['volume'].apply(lambda x: f"{x:,.0f} shares")
    
    print(forecast_df.to_markdown(numalign="left", stralign="left"))


# --- Main Execution Block for Bayesian Optimization (Unchanged) ---
if __name__ == '__main__':
    try:
        # 1. Data Prep
        stock_df = load_and_clean_data(file_path='data/PLX_price_history_with_indicators.csv')
        X, y, full_scaler, target_scaler, final_df_features = preprocess_data(
            stock_df, TIME_STEP, FEATURES, TARGET_COLUMNS
        )
        X_train, X_test, y_train, y_test = split_data(
            X, y, TRAIN_SPLIT_PERCENT
        )

        # 2. Cleanup 
        if os.path.exists(TUNER_DIR):
            shutil.rmtree(TUNER_DIR)
            print(f"🗑️ Previous tuning results cleared from {TUNER_DIR}")

        # 3. Instantiate the Bayesian Optimization Tuner
        tuner = kt.BayesianOptimization(
            build_hypermodel,
            objective=OBJECTIVE_METRIC,
            max_trials=MAX_TRIALS,
            executions_per_trial=EXECUTIONS_PER_TRIAL,
            directory=TUNER_DIR,
            project_name=PROJECT_NAME,
            overwrite=True
        )

        print("\n🚀 Starting Bayesian Optimization Hyperparameter Search...")
        print(f"Total Trials: {MAX_TRIALS}, Objective: {OBJECTIVE_METRIC}")
        
        # 4. Run the Search
        tuner.search(
            X_train, y_train,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            validation_split=VALIDATION_SPLIT,
            verbose=1,
        )

        # 5. Get the Best Model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        print("\n" + "="*50)
        print("✨ **Bayesian Optimization Complete** ✨")
        print("--- Best Hyperparameters Found ---")
        print(f"LSTM Layers: {best_hps.get('num_layers')}")
        print(f"Units per Layer: {best_hps.get('units')}")
        print(f"Dropout Rate: {best_hps.get('dropout_rate'):.2f}")
        print(f"Learning Rate: {best_hps.get('learning_rate'):.5f}")
        print(f"Best Validation Loss ({OBJECTIVE_METRIC}): {tuner.oracle.get_best_trials()[0].score:.4f}")
        print("="*50)

        # 6. Final Evaluation and Prediction using the Best Model
        evaluate_predict_and_forecast(
            best_model, X_test, y_test, target_scaler, final_df_features, full_scaler
        )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")