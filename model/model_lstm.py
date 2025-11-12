from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Local imports
from configs import *

class ModelLstm():
    def __init__(self):
        pass

    def build_model(self, hp):
        model = Sequential()
        
        # Hyperparameters
        hp_layers = hp.Int('num_layers', min_value=1, max_value=MAX_LAYER, step=1)
        hp_units = hp.Int('units', min_value=MIN_NEURONS_LAYER, max_value=MAX_NEURONS_LAYER, step=STEP_NEURONS_LAYER)
        hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        
        # Input Shape depends on the model type
        n_features = len(FEATURES)
        input_shape_layer = (TIME_STEP, n_features)

        # --- Build the model ---
        
        for i in range(hp_layers):
            return_seq = (i < hp_layers - 1)
            if i == 0:
                model.add(LSTM(units=hp_units, return_sequences=return_seq, 
                            input_shape=input_shape_layer))
            else:
                model.add(LSTM(units=hp_units, return_sequences=return_seq))

            # Add Dropout after the layer
            if return_seq:
                model.add(Dropout(hp_dropout))
            
        # Final Dense Layer
        n_targets = len(TARGET_COLUMNS)
        model.add(Dense(units=n_targets, activation=ACTIVATION)) 
        
        # Compilation
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mean_squared_error',
            metrics=['mse'] 
        )
        
        return model