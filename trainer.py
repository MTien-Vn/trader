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
import joblib # Import joblib for saving scalers

# Local imports
from config import *
from data_processor import  preprocess_data
from model import build_model

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

def save_scalers(full_scaler, target_scaler):
    """Saves the fitted scalers using joblib."""
    os.makedirs(os.path.dirname(FULL_SCALER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TARGET_SCALER_PATH), exist_ok=True)
    
    try:
        joblib.dump(full_scaler, FULL_SCALER_PATH)
        print(f"✅ Scalers saved to {os.path.dirname(FULL_SCALER_PATH)}")

        joblib.dump(target_scaler, TARGET_SCALER_PATH)
        print(f"✅ Scalers saved to {os.path.dirname(TARGET_SCALER_PATH)}")
    except Exception as e:
        print(f"❌ Error saving scalers: {e}")

def save_best_model(model):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    save_model(model, MODEL_SAVE_PATH)
    print("✅ Model saved successfully.")

def get_best_model(tuner):
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
    return best_model

# --- Main Execution ---

if __name__ == '__main__':
    
    print(f"🚀 Training with Architecture: **{SELECTED_MODEL_TYPE}**")
    
    # 1. Data Prep
    X_train, X_test, y_train, y_test, full_scaler, target_scaler, final_df_features = preprocess_data(
        DATA_TRAINING_FILE_PATH, TIME_STEP, FEATURES, TARGET_COLUMNS
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
    best_model = get_best_model(tuner)

    save_best_model(best_model)

    save_scalers(full_scaler, target_scaler)

    # 7. Final Evaluation and Prediction using the Best Model
    evaluate(
        best_model, X_test, y_test, target_scaler, SELECTED_MODEL_TYPE
    )
        
