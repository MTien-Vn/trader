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

# --- Global Hyperparameters (Fixed) ---
TIME_STEP = 20 
TRAIN_SPLIT_PERCENT = 0.8
FEATURES = ['volume', 'close', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
TARGET_COLUMNS = ['close', 'volume'] 
VALIDATION_SPLIT = 0.1
ACTIVATION = 'linear'
OPTIMIZER = 'adam'
MAX_LAYER = 3
MAX_NEURONS_LAYER = 256
MIN_NEURONS_LAYER = 32
STEP_NEURONS_LAYER = 32
EPOCHS = 50
BATCH_SIZE = 32

# --- KerasTuner Hyperparameter Search Settings ---
MAX_TRIALS = 15     # Max number of hyperparameter combinations to try
EXECUTIONS_PER_TRIAL = 1 # Number of models to train for each trial (1 is usually fine)
OBJECTIVE_METRIC = 'val_loss' # The metric to minimize (Validation Loss)
PROJECT_NAME = 'lstm_stock_bo'
TUNER_DIR = 'tuning_results'

# --- Data Loading and Preprocessing Functions (Unchanged) ---

def load_and_clean_data(file_path='stock_data.csv'):
    """Loads and performs basic cleaning/validation on the stock data."""
    # Load the data (assuming the index is already set or handled as a column)
    df = pd.read_csv(file_path, parse_dates=True)
    # 1. Convert 'date' and sort data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    # 2. Check for features availability
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")
    
    # 3. Handle Missing Values (Imputation/Deletion)
    df.dropna(subset=TARGET_COLUMNS, inplace=True)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].mean())
    
    if df[FEATURES].isnull().values.any():
         print("⚠️ WARNING: NaNs still present after imputation. Dropping remaining rows with NaNs.")
         df.dropna(subset=FEATURES, inplace=True)

    # 4. Drop the date column for modeling
    df = df.drop(columns=['date'])

    print(f"✅ Data Loaded and Cleaned. Total rows: {len(df)}. Using features: {FEATURES}")
    return df

def create_sequences(data, time_step, features, target_columns):
    """
    Creates the time-stepped sequences (X) and next-day targets (Y) 
    for the LSTM model.
    """
    X, Y = [], []
    
    # Get the indices of the target columns within the full features list
    target_indices = [features.index(col) for col in target_columns]
    
    for i in range(len(data) - time_step):
        # Input sequence (X)
        X.append(data[i:(i + time_step)])
        
        # Output target (Y): next day's target values
        Y.append(data[i + time_step, target_indices]) 
        
    return np.array(X), np.array(Y)

def preprocess_data(df, time_step, features, target_columns):
    """
    Handles scaling and sequence creation.
    Returns: X, y, full_data_scaler, target_scaler, final_df_features
    """
    # 1. Scale the entire dataset
    full_data_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = full_data_scaler.fit_transform(df[features].values)

    # 2. Create a separate scaler for the target columns for inverse transformation
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit_transform(df[target_columns].values)
    
    # 3. Create sequences
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

# --- Hypermodel Function (The key change for KerasTuner) ---

def build_hypermodel(hp):
    """
    Defines the LSTM model structure and the hyperparameter search space.
    This function is passed to the KerasTuner.
    """
    model = Sequential()
    
    # Define the search space for the number of LSTM layers (1 to 3)
    # The default for the first layer (return_sequences=True) is handled in the loop logic below
    hp_layers = hp.Int('num_layers', min_value=1, max_value=MAX_LAYER, step=1)
    
    # Define the search space for the number of neurons in each layer (e.g., 32 to 256, step 32)
    hp_units = hp.Int('units', min_value=MIN_NEURONS_LAYER, max_value=MAX_NEURONS_LAYER, step=STEP_NEURONS_LAYER)
    
    # Define the search space for dropout rate (e.g., 0.1 to 0.5, step 0.1)
    hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    # --- Build the Stacked LSTM Model Dynamically ---
    for i in range(hp_layers):
        
        # return_sequences must be True for all layers except the last one
        return_seq = (i < hp_layers - 1)
        
        if i == 0:
            # First layer needs input_shape
            model.add(LSTM(units=hp_units, return_sequences=return_seq, 
                           input_shape=(TIME_STEP, X_train.shape[2])))
        else:
            # Subsequent layers use the same unit count
            model.add(LSTM(units=hp_units, return_sequences=return_seq))
            
        model.add(Dropout(hp_dropout))
        
    # Output Layer: Dense layer for n_targets (Multivariate Output)
    n_targets = y_train.shape[1]
    model.add(Dense(units=n_targets, activation=ACTIVATION)) 
    
    # Tune the learning rate for the Adam optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error',
        metrics=['mse'] # KerasTuner will optimize based on the 'val_loss' from the loss function
    )
    
    return model

def evaluate_and_predict(model, X_test, y_test, target_scaler, last_data_df, full_data_scaler):
    """
    Evaluates the model on the test set and makes a final single-day prediction.
    """
    
    # --- A. Test Evaluation ---
    scaled_predictions = model.predict(X_test, verbose=0)

    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(scaled_predictions)
    y_test_original = target_scaler.inverse_transform(y_test)

    actual_close = y_test_original[:, 0]
    predicted_close = predictions[:, 0]
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
    print(f"\n--- Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE) for Close Price on Test Set: ${rmse:,.2f}")

    # --- B. Next Day Prediction ---
    
    # 1. Select and scale the last required sequence
    last_sequence = last_data_df.tail(TIME_STEP).values
    scaled_last_sequence = full_data_scaler.transform(last_sequence)

    # 2. Reshape to 3D for model input: (1 sample, TIME_STEP, n_features)
    X_next = scaled_last_sequence.reshape(1, TIME_STEP, last_data_df.shape[1])

    # 3. Predict and Inverse Transform
    scaled_prediction = model.predict(X_next, verbose=0)
    final_prediction = target_scaler.inverse_transform(scaled_prediction)

    # Extract results
    predicted_next_close = final_prediction[0, 0]
    predicted_next_volume = final_prediction[0, 1]

    print("\n--- Next Day Prediction ---")
    print(f"Predicted Close Price: {predicted_next_close:,.2f} VND")
    print(f"Predicted Trading Volume: {predicted_next_volume:,.0f} shares")

    return predictions

# --- Main Execution Block for Bayesian Optimization ---
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

        # 2. Cleanup (Optional, but useful for fresh runs)
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
        # EPOCHS and BATCH_SIZE are now passed directly to the search method
        # Note: You can tune EPOCHS and BATCH_SIZE within the search, but they are often fixed here.
        tuner.search(
            X_train, y_train,
            epochs=EPOCHS, # Fixed epochs for tuning, can be made a hyperparameter if needed
            batch_size=BATCH_SIZE, # Fixed batch size
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
        evaluate_and_predict(
            best_model, X_test, y_test, target_scaler, final_df_features, full_scaler
        )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")