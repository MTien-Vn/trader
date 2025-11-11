# model_trainer.py

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import kerastuner as kt
import os 
import shutil
import copy 
from tensorflow.keras.models import save_model

# Local imports
from config import *
from data_processor import load_and_clean_data, preprocess_data, get_target_indices
from indicator_utils import recalculate_non_target_features_production
from model import build_model


# --- Multi-Step Forecasting (Moved from main block) ---

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


def forecast(model, target_scaler, last_data_df, full_data_scaler, model_type, stock_symbol):
    # 1. Fetch the necessary extended history for indicators
    last_sequence_original = last_data_df.tail(TIME_STEP).values
    scaled_last_sequence = full_data_scaler.transform(last_sequence_original)
    
    # 2. Perform the recursive forecast
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
    forecast_df.index = [f"Day +{i+1}" for i in range(FORECAST_DAYS)]
    
    forecast_df['PCT_CHANGE'] = forecast_df['PCT_CHANGE'].apply(lambda x: f"{x:,.2f}")
    
    print(forecast_df.to_markdown(numalign="left", stralign="left"))

def evaluate(model, X_test, y_test, target_scaler, model_type):
    if model_type == 'ConvLSTM':
        # Reshape X_test for ConvLSTM (samples, time_steps, 1, 1, features)
        X_test_reshaped = X_test.reshape(X_test.shape[0], TIME_STEP, 1, 1, X_test.shape[-1])
    else:
        X_test_reshaped = X_test
        
    scaled_predictions = model.predict(X_test_reshaped, verbose=0)
    predictions = target_scaler.inverse_transform(scaled_predictions)
    y_test_original = target_scaler.inverse_transform(y_test)
    actual_close = y_test_original[:, 0]
    predicted_close = predictions[:, 0]
    rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
    print(f"\n--- Evaluation Results (Test Set) ---")
    print(f"Root Mean Squared Error (RMSE) for Close Price: ${rmse:,.2f}")


# --- Main Execution ---

if __name__ == '__main__':
    
    # --- CHOOSE MODEL HERE: 'LSTM', 'Bi-LSTM', or 'ConvLSTM' ---
    SELECTED_MODEL_TYPE = 'LSTM' # <--- Change this line to switch models!
    print(f"🚀 Training with Architecture: **{SELECTED_MODEL_TYPE}**")
    
    # 1. Data Prep
    X_train, X_test, y_train, y_test, full_scaler, target_scaler, final_df_features = preprocess_data(
        DATA_FILE_PATH, TIME_STEP, FEATURES, TARGET_COLUMNS
    )

    # 2. Setup Callbacks (Early Stopping & LR Scheduling)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=OBJECTIVE_METRIC, 
            patience=PATIENCE_EARLY_STOPPING, 
            mode='min', 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=OBJECTIVE_METRIC,
            factor=0.5,
            patience=PATIENCE_LR_SCHEDULING,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
    ]
    
    # 3. Cleanup 
    turner_dir = f"{TUNER_DIR}_{SELECTED_MODEL_TYPE}"
    if os.path.exists(turner_dir):
        shutil.rmtree(turner_dir)
        print(f"🗑️ Previous tuning results cleared from {turner_dir}")

    # 4. Instantiate the Tuner (Pass the selected model type)
    hypermodel_fn = lambda hp: build_model(hp, model_type=SELECTED_MODEL_TYPE)
    
    tuner = kt.BayesianOptimization(
        hypermodel_fn,
        objective=OBJECTIVE_METRIC,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=turner_dir,
        project_name=PROJECT_NAME,
        overwrite=True
    )

    print("\n🚀 Starting Bayesian Optimization Hyperparameter Search...")
    
    # 5. Run the Search
    # ConvLSTM input must be reshaped for KerasTuner search
    if SELECTED_MODEL_TYPE == 'ConvLSTM':
        X_train_tuner = X_train.reshape(X_train.shape[0], TIME_STEP, 1, 1, X_train.shape[-1])
    else:
        X_train_tuner = X_train

    tuner.search(
        X_train_tuner, y_train,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Get the Best Model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print("\n" + "="*50)
    print("✨ **Bayesian Optimization Complete** ✨")
    print("--- Best Hyperparameters Found ---")
    print(f"Architecture: {SELECTED_MODEL_TYPE}")
    print(f"LSTM/ConvLSTM Layers: {best_hps.get('num_layers')}")
    print(f"Units per Layer: {best_hps.get('units')}")
    if SELECTED_MODEL_TYPE == 'ConvLSTM':
        print(f"Conv Filters: {best_hps.get(f'conv_filters_0')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate'):.2f}")
    print(f"Learning Rate: {best_hps.get('learning_rate'):.5f}")
    print("="*50)

    # # --- Global Setting for Model Save Path ---
    # MODEL_SAVE_PATH = os.path.join('best_model', f'{PROJECT_NAME}_{SELECTED_MODEL_TYPE}.keras')
    # os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # save_model(best_model, MODEL_SAVE_PATH)
    # print("✅ Model saved successfully.")

    # 7. Final Evaluation and Prediction using the Best Model
    evaluate(
        best_model, X_test, y_test, target_scaler, SELECTED_MODEL_TYPE
    )

    for stock_symbol in PREDICT_STOCK_SYMBOLS:
        filename = f"{stock_symbol}_price_history.csv"
        file_path = os.path.join(FOLDER_PATH, filename)

        df = pd.read_csv(file_path, parse_dates=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        df_features = df[FEATURES]

        forecast(
            best_model, target_scaler, df_features, full_scaler, SELECTED_MODEL_TYPE, stock_symbol
        )
        
