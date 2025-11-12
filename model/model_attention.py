from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Layer # Base layer for custom block
from tensorflow import keras

from configs import *

class TransformerBlock(Layer):
    """
    A single Transformer Encoder block consisting of Multi-Head Attention and a Feed-Forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training):
        # 1. Self-Attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # Add & Norm
        out1 = self.layernorm1(inputs + attn_output)
        
        # 2. Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Add & Norm
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class ModelAttention():
    def __init__(self):
        pass

    def build_model(self, hp):
        """
        Defines and compiles the Self-Attention (Transformer Encoder) model.
        
        """

        # Define tunable hyperparameters
        num_blocks = hp.Int('n_blocks', min_value=1, max_value=MAX_LAYER, step=1)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        ff_dim = hp.Int('ff_dim', min_value=MIN_NEURONS_LAYER, max_value=MAX_NEURONS_LAYER, step=STEP_NEURONS_LAYER)
        dense_units = hp.Int('dense_units', min_value=10, max_value=64, step=10)
        dropout_rate = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3])
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

        # The input layer accepts the sequence (e.g., 60 days of data)
        n_features = len(FEATURES)
        input_shape_layer = (TIME_STEP, n_features)
        inputs = Input(shape=input_shape_layer)
        
        # Since the input features are already numerical, we treat the feature count 
        # as our embedding dimension (or project it to a desired size if needed, 
        # but using the feature count is a common time-series practice).
        
        x = inputs
        
        # Stack multiple Transformer Blocks
        for _ in range(num_blocks):
            x = TransformerBlock(
                embed_dim=n_features, 
                num_heads=num_heads, 
                ff_dim=ff_dim, 
                rate=dropout_rate
            )(x)
            
        # Pool the output across the sequence length (60 days) to get a single vector 
        # representing the entire sequence context.
        x = GlobalAveragePooling1D()(x)
        
        # Final dense layers for prediction
        n_targets = len(TARGET_COLUMNS)
        outputs = Dense(n_targets, activation="relu")(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Use a tunable learning rate for the Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Using Adam optimizer and Mean Squared Error loss (standard for regression)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        return model
