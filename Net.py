import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LSTM, GRU, Flatten,RNN,SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def lstm_3_net(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = LSTM(128, return_sequences=True)(input)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model

def lstm_3_net_NOBN(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = LSTM(128, return_sequences=True)(input)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model

def rf_lstm(shape=(30, 3)):
    input = Input(shape, name='self_input1')
    x = LSTM(30)(input)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    model = Model(input, x)
    # model.summary()
    return model

def rnn_3_net(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = SimpleRNN(128, return_sequences=True)(input)
    x = SimpleRNN(64, return_sequences=True)(x)
    x = SimpleRNN(32, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model

def gru_3_net(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = GRU(128, return_sequences=True)(input)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(32, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model

def lstm_2_net(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = LSTM(128, return_sequences=True)(input)
    x = LSTM(64, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model


def lstm_1_net(shape=(60, 10)):
    input = Input(shape, name='self_input1')
    x = LSTM(128)(input)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model


def bp_5_net(shape=(30, 10)):
    input = Input(shape, name='self_input1')
    x = Flatten()(input)
    x = Dense(1000, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1)(x)
    model = Model(input, x)
    # model.summary()
    return model


if __name__ == '__main__':
    lstm_3_net()
