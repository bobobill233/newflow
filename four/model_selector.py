from model import CNN_LSTM_GAM, shijie, Swin_CNN, Swin_CNN_LSTM, Swin_CNN_LSTM_GAM, Swin_CNN_LSTM_SelfAttention


def get_model(model_name):
    if model_name == 'CNN_LSTM_GAM':
        return CNN_LSTM_GAM.CNNLSTMGAM()
    elif model_name == 'shijie':
        return shijie.CNNLSTM()
    elif model_name == 'Swin_CNN':
        return Swin_CNN.SwinCNN()
    elif model_name == 'Swin_CNN_LSTM':
        return Swin_CNN_LSTM.SwinCNNLSTM()
    elif model_name == 'Swin_CNN_LSTM_GAM':
        return Swin_CNN_LSTM_GAM.SwinCNNLSTMGAM()
    elif model_name == 'Swin_CNN_LSTM_SelfAttention':
        return Swin_CNN_LSTM_SelfAttention.SwinCNNLSTMSelfAttention()
    else:
        raise ValueError("Unknown model name")
