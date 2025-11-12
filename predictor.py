# predictor.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib # Import joblib for loading scalers
import os
import copy
from datetime import datetime

# --- Import configurations and utility functions ---
from config import (
    FEATURES, FOLDER_PREDICT_PATH, FOLDER_RESULT_PATH, FULL_SCALER_PATH, MODEL_SAVE_PATH, PREDICT_STOCK_SYMBOLS, SELECTED_MODEL_TYPE, TARGET_COLUMNS, NON_TARGET_FEATURES, TARGET_SCALER_PATH, TIME_STEP, 
    FORECAST_DAYS
)
from data_processor import get_target_indices
# ...


def predict_next_n_days(model, initial_scaled_sequence, full_data_scaler, target_scaler, n_days, model_type='LSTM'):
    """
    Performs recursive multi-step forecasting for n_days using 
    recalculated indicators.
    """
    current_scaled_sequence = copy.deepcopy(initial_scaled_sequence)
    # current_extended_data_history = copy.deepcopy(extended_data_history) 
    forecasted_predictions = []
    
    target_indices = get_target_indices(FEATURES, TARGET_COLUMNS)
    non_target_indices = [FEATURES.index(col) for col in NON_TARGET_FEATURES]

    # Handle shape based on model type
    if model_type == 'ConvLSTM':
        # Input shape must be (1, TIME_STEP, 1, 1, features)
        input_shape_fn = lambda seq: seq.reshape(1, TIME_STEP, 1, 1, len(FEATURES))
    else:
        # Input shape must be (1, TIME_STEP, features)
        input_shape_fn = lambda seq: seq.reshape(1, TIME_STEP, len(FEATURES))


    for day in range(n_days):
        
        # 1. Predict the next step (Close, Volume) - result is scaled
        X_next = input_shape_fn(current_scaled_sequence)
        scaled_prediction = model.predict(X_next, verbose=0)
        
        # 2. Inverse transform the prediction to get unscaled Close/Volume
        final_prediction_unscaled = target_scaler.inverse_transform(scaled_prediction)[0]
        # Map the unscaled array to a dictionary for easy access
        next_predictions = {
            col: final_prediction_unscaled[i] 
            for i, col in enumerate(TARGET_COLUMNS)
        }

        # 3. Recalculate non-target features (unscaled)
        # next_non_target_unscaled = recalculate_non_target_features_production(
        #     current_extended_data_history, next_predictions
        # )

        # 4. Construct the complete new point (unscaled)
        new_unscaled_data_point = np.zeros(len(FEATURES))
        # Place the 4 target predictions into the new point
        for i, col in enumerate(TARGET_COLUMNS):
             new_unscaled_data_point[target_indices[i]] = next_predictions[col]

        # Place the 7 calculated non-target features into the new point
        # new_unscaled_data_point[non_target_indices] = next_non_target_unscaled
        
        # 5. Scale the new point
        new_scaled_data_point = full_data_scaler.transform(new_unscaled_data_point.reshape(1, -1))[0]

        # 6. Update the sliding window sequences
        current_scaled_sequence = np.vstack([current_scaled_sequence[1:], new_scaled_data_point])

        # New row as a dictionary for reliable DataFrame conversion
        # new_data = {
        #     'close': [next_predictions['close']],
        #     'high': [next_predictions['high']],
        #     'low': [next_predictions['low']]
        # }
        # new_row = pd.DataFrame(new_data)

        # # Drop the oldest row (index 0 after reset) and append the newest prediction
        # current_extended_data_history = pd.concat(
        #     [current_extended_data_history.iloc[1:].reset_index(drop=True), new_row], 
        #     ignore_index=True
        # )
        
        # 7. Store the unscaled prediction
        forecasted_predictions.append(final_prediction_unscaled)
        
    return np.array(forecasted_predictions)


def forecast(model, target_scaler, scaled_last_sequence, full_data_scaler, model_type, stock_symbol):
    forecasted_results = predict_next_n_days(
        model, 
        scaled_last_sequence, 
        full_data_scaler, 
        target_scaler, 
        n_days=FORECAST_DAYS,
        model_type=model_type
    )

    print(f"\n--- {FORECAST_DAYS}-Day Multi-Step Forecast: {stock_symbol} ---")
    
    forecast_df = pd.DataFrame(forecasted_results, columns=TARGET_COLUMNS)
    forecast_df["Day"] = [f"Day +{i+1}" for i in range(FORECAST_DAYS)]
    forecast_df = forecast_df[["Day"] + [col for col in forecast_df.columns if col != "Day"]]
    
    # Round numeric values
    forecast_df = forecast_df.round(2)
    
    print(forecast_df.to_markdown(numalign="left", stralign="left"))

    return forecast_df

def load_data_and_scalers(new_data_file_path):
    """
    Loads historical data, loads the saved scalers, and prepares the 
    final input sequence.
    """
    try:
        # 1. Load Scalers
        full_scaler = joblib.load(FULL_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        print("✅ Scalers loaded successfully.")

        # 2. Load and clean the newest data
        df = pd.read_csv(new_data_file_path, parse_dates=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        # We only need the latest data slice, so ensure we have enough rows
        df = df.tail(TIME_STEP).dropna(subset=FEATURES) 
        df = df.drop(columns=['date'])
        
        # 3. Prepare prediction input
        full_data_for_prediction = df[FEATURES].values
        
        # Scale the data using the loaded scaler
        scaled_full_data = full_scaler.transform(full_data_for_prediction)
        
        # The input sequence for the model is the last TIME_STEP days
        initial_scaled_sequence = scaled_full_data[-TIME_STEP:]
        
        return initial_scaled_sequence, full_scaler, target_scaler
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found ({e}). Check data/scaler paths.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error during loading or data preparation: {e}")
        return None, None, None, None


if __name__ == '__main__':
    try:
        final_model = load_model(MODEL_SAVE_PATH)
        print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ FAILED to load model. Please ensure the path is correct and the file exists. Error: {e}")
        exit()

    results_summary = []  # Store all stock results

    # Ensure the folder exists
    current_date = datetime.now().strftime("%Y-%m-%d")
    folder_result = f'{FOLDER_RESULT_PATH}_{current_date}'
    os.makedirs(folder_result, exist_ok=True)

    for stock_symbol in PREDICT_STOCK_SYMBOLS:
        filename = f"{stock_symbol}_price_history.csv"
        file_path = os.path.join(FOLDER_PREDICT_PATH, filename)

        scaled_seq, full_scaler, target_scaler = load_data_and_scalers(file_path)

        if scaled_seq is None:
            exit()

        forecasted_df = forecast(
            final_model, target_scaler, scaled_seq, full_scaler, SELECTED_MODEL_TYPE, stock_symbol
        )

        # Compute total % change
        total_pct_change = forecasted_df['PCT_CHANGE'].sum()

        # Add to results summary
        results_summary.append({
            "symbol": stock_symbol,
            "total_pct_change": total_pct_change,
        })

        output_csv = os.path.join(folder_result, f"{stock_symbol}_forecast_summary.csv")
        forecasted_df.to_csv(output_csv, index=False)

    # --- After loop: build combined result CSV ---
    summary_df = pd.DataFrame(results_summary)
    
    # Sort descending by total % change
    summary_df = summary_df.sort_values(by="total_pct_change", ascending=False).reset_index(drop=True)

    # Save to CSV
    output_csv = os.path.join(folder_result, f"forecast_summary.csv")
    summary_df.to_csv(output_csv, index=False)

    print(f"\n✅ Saved forecast summary to {output_csv}")
    print(summary_df[["symbol", "total_pct_change"]])
