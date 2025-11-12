# Local imports
from configs import *
from model import *

def get_model(hp, model_type='LSTM'):
    """
    Defines the LSTM, Bi-LSTM, or ConvLSTM model structure and the 
    hyperparameter search space.
    
    Args:
        hp (kt.HyperParameters): The hyperparameters object.
        model_type (str): 'LSTM', 'Bi-LSTM', or 'ConvLSTM'.
    """
    if model_type == 'attention':
        return ModelAttention().build_model(hp)
    elif model_type == 'Bi-LSTM':
        return ModelBiLstm().build_model(hp)
    elif model_type == 'ConvLSTM':
        return ModelConvLstm().build_model(hp)
    elif model_type == 'LSTM':
        return ModelLstm().build_model(hp)
    elif model_type == 'GRU':
        return ModelGru().build_model(hp) 
    