from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ConvLSTM2D

# Local imports
from configs import *

class ModelConvLstm():
    def __init__(self):
          pass

    def build_model(self, hp):
        
        model = Sequential()
        
        # Hyperparameters
        hp_layers = hp.Int('num_layers', min_value=1, max_value=MAX_LAYER, step=1)
        
        # Input Shape depends on the model type
        n_features = len(FEATURES)
        input_shape_layer = (TIME_STEP, 1, 1, n_features) 

        # --- Build the model ---
        
        for i in range(hp_layers):
            return_seq = (i < hp_layers - 1)
            
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