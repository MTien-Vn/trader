from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, ConvLSTM2D
import kerastuner as kt

# Local imports
from config import *

def build_model(hp, model_type='LSTM'):
    """
    Defines the LSTM, Bi-LSTM, or ConvLSTM model structure and the 
    hyperparameter search space.
    
    Args:
        hp (kt.HyperParameters): The hyperparameters object.
        model_type (str): 'LSTM', 'Bi-LSTM', or 'ConvLSTM'.
    """
    model = Sequential()
    
    # Hyperparameters
    hp_layers = hp.Int('num_layers', min_value=1, max_value=MAX_LAYER, step=1)
    hp_units = hp.Int('units', min_value=MIN_NEURONS_LAYER, max_value=MAX_NEURONS_LAYER, step=STEP_NEURONS_LAYER)
    hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    # Input Shape depends on the model type
    n_features = len(FEATURES)
    if model_type == 'ConvLSTM':
        # ConvLSTM requires a 5D input: (samples, time_steps, rows, cols, features)
        # We treat TIME_STEP as the time sequence, and the feature vector as (1, features) spatial
        input_shape_layer = (TIME_STEP, 1, 1, n_features) 
    else:
        # LSTM/Bi-LSTM require a 3D input: (samples, time_steps, features)
        input_shape_layer = (TIME_STEP, n_features)

    # --- Build the model ---
    
    for i in range(hp_layers):
        return_seq = (i < hp_layers - 1)
        
        if model_type == 'ConvLSTM':
            # ConvLSTM Layer - Use the entire sequence length for the outer dimension
            hp_filters = hp.Choice(f'conv_filters_{i}', values=CONV_FILTER_POOL)
            
            if i == 0:
                # The first ConvLSTM needs a special input shape defined by (timesteps, rows, cols, channels)
                model.add(ConvLSTM2D(
                    filters=hp_filters, 
                    kernel_size=CONV_KERNEL_SIZE,
                    padding='same',
                    return_sequences=return_seq, 
                    input_shape=input_shape_layer
                ))
            else:
                model.add(ConvLSTM2D(
                    filters=hp_filters, 
                    kernel_size=CONV_KERNEL_SIZE,
                    padding='same',
                    return_sequences=return_seq
                ))
            # Reshape needed only if return_sequences=False is reached
            if not return_seq and hp_layers > 1:
                 # Flatten to 2D before Dropout/Dense if not returning a sequence
                 model.add(keras.layers.Flatten())
                 
        elif model_type == 'Bi-LSTM':
             # Bi-LSTM Layer
             if i == 0:
                 layer = Bidirectional(LSTM(units=hp_units, return_sequences=return_seq), 
                                        input_shape=input_shape_layer)
             else:
                 layer = Bidirectional(LSTM(units=hp_units, return_sequences=return_seq))
             model.add(layer)
             
        else: # Default 'LSTM'
             if i == 0:
                 model.add(LSTM(units=hp_units, return_sequences=return_seq, 
                                input_shape=input_shape_layer))
             else:
                 model.add(LSTM(units=hp_units, return_sequences=return_seq))

        # Add Dropout after the layer
        if return_seq or model_type != 'ConvLSTM': # Dropout is typically 2D after ConvLSTM output
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