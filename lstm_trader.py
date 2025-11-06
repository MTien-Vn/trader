import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# --- Global Hyperparameters ---
TIME_STEP = 60 
TRAIN_SPLIT_PERCENT = 0.8
FEATURES = ['close', 'volume', 'open', 'high', 'low'] 
TARGET_COLUMNS = ['close', 'volume'] 
EPOCHS = 50
BATCH_SIZE = 32
DROP_RATE = 0.3
VALIDATION_SPLIT = 0.1
NEURON_EACH_LAYER = 100
ACTIVATION = 'linear'
OPTIMIZER ='adam'
 
def load_and_clean_data(file_path='stock_data.csv'):
    """Loads and performs basic cleaning/validation on the stock data."""
    # Load the data (assuming the index is already set or handled as a column)
    df = pd.read_csv(file_path, parse_dates=True)
    
    # Simple data validation for OHLC integrity
    df['high'] = df[FEATURES].filter(regex='high|close|open').max(axis=1)
    df['low'] = df[FEATURES].filter(regex='low|close|open').min(axis=1)

    # Check for features availability
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")

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

def build_and_train_model(X_train, y_train, epochs, batch_size):
    """Defines, compiles, and trains the Stacked LSTM model."""
    
    n_features = X_train.shape[2]
    n_targets = y_train.shape[1]

    model = Sequential()

    # Layer 1: LSTM with return_sequences=True (Stacked LSTM)
    model.add(LSTM(units=NEURON_EACH_LAYER, return_sequences=True, 
                   input_shape=(TIME_STEP, n_features)))
    model.add(Dropout(DROP_RATE)) 

    # Layer 2: Final LSTM layer
    model.add(LSTM(units=NEURON_EACH_LAYER, return_sequences=False))
    model.add(Dropout(DROP_RATE))

    # Output Layer: Dense layer for n_targets (Multivariate Output)
    model.add(Dense(units=n_targets, activation=ACTIVATION)) 

    # Compile
    model.compile(optimizer=OPTIMIZER, loss='mean_squared_error')

    print("\n--- Model Summary ---")
    model.summary()
    
    print("\n🚀 Starting Model Training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,          
        batch_size=batch_size,      
        validation_split=VALIDATION_SPLIT, 
        verbose=0 # Set to 1 for full progress bar
    )
    print("✅ Training Complete.")
    
    return model, history

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
    print(f"Root Mean Squared Error (RMSE) for Close Price: ${rmse:,.2f}")

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

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        # 1. Load and Clean Data
        stock_df = load_and_clean_data(file_path='data/VIX_price_history.csv')

        # 2. Preprocess Data (Scale and Create Sequences)
        X, y, full_scaler, target_scaler, final_df_features = preprocess_data(
            stock_df, TIME_STEP, FEATURES, TARGET_COLUMNS
        )

        # 3. Split Data
        X_train, X_test, y_train, y_test = split_data(
            X, y, TRAIN_SPLIT_PERCENT
        )

        # 4. Build and Train Model
        model, history = build_and_train_model(
            X_train, y_train, EPOCHS, BATCH_SIZE
        )
        
        # 5. Evaluate and Predict
        predictions = evaluate_and_predict(
            model, X_test, y_test, target_scaler, final_df_features, full_scaler
        )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")