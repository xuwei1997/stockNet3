#在模型上使用不同的模型+不同的损失函数预测股票涨跌额

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from Net import *
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.metrics import BinaryAccuracy, binary_accuracy
from StockLossAndMetrics import metrics, loss_nloss, loss_nlossbc, loss_mbe
import random
import os
from sliding_window import sliding_window_view
# import tushare as ts
# from nbeats_keras.model import NBeatsNet as NBeatsKeras
from nbeats_model import NBeatsNet as NBeatsKeras
import datetime


def max_index_max(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return max(index)  # 返回一个列表


class trainStockNet(object):
    def __init__(self, ticker, loss_fun, net_model, epochs):
        self.ticker = ticker
        self.loss_fun = loss_fun
        self.net_model = net_model
        self.epochs = epochs

    def data_preprocessing(self, kind=0, windows=50):
        # print('data_preprocessing')
        # kind 0-涨跌 1-收盘价
        # 取数据

        df1 = pd.read_excel('./data/d' + str(self.ticker) + '.xlsx')
        df2 = pd.read_csv('./data/m' + str(self.ticker) + '.csv')

        df = pd.merge(df1, df2, how='inner', on='tradeDate')
        # df = df.iloc[::-1].reset_index(drop=True)
        # print(df)
        # print(df.isnull().values.any())
        df = df.fillna(0)
        # print(df.isnull().values.any())

        key = ['preClosePrice', 'actPreClosePrice', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice',
               'turnoverVol', 'turnoverValue', 'dealAmount', 'turnoverRate', 'accumAdjFactor', 'negMarketValue',
               'marketValue', 'chgPct', 'PE_x', 'PE1', 'PB_x', 'vwap'] + df2.columns.tolist()[4:]
        # print(key.index('PE'))
        key.pop(key.index('PE'))
        # print(key)
        key.pop(key.index('PB'))
        # key = key.remove('PB')

        # key = ['preClosePrice', 'actPreClosePrice', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice',
        #        'turnoverVol', 'turnoverValue', 'dealAmount', 'turnoverRate', 'accumAdjFactor', 'negMarketValue',
        #        'marketValue', 'chgPct', 'PE_x', 'PE1', 'PB_x', 'vwap']
        # print(key)
        # global k_long
        self.k_long = len(key)

        # print(k_long)
        # 去掉了时间维度！

        x = np.array(df[key])
        # y = np.array(df[['openPrice']]) - np.array(df[['preClosePrice']])
        y = np.array(df[['closePrice']]) - np.array(df[['preClosePrice']])

        if kind == 0:  # 涨跌额
            y_max = abs(max(y.min(), y.max(), key=abs))
            # # 最大值
            # print('max:')
            # print(y_max)

            # y 归一化
            y = y / y_max

            # 归一化
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x = x_scaler.fit_transform(x)

            # y 回归正确的归一化
            # x[:, 5] = y.reshape((-1))
            y_min = 0
            # print(x.max())

        else:  # 收盘价
            pass

        # 滑动窗口
        x = sliding_window_view(x, (windows, self.k_long)).reshape((-1, windows, len(key)))
        # print('sliding_window_view')
        # print(x.shape)
        # print(y.shape)
        x = x[0:-1]
        y = y[windows:]
        # print(x.shape)
        # print(y.shape)

        self.x = x
        self.y = y
        self.y_max = y_max
        self.y_min = y_min

        self.scaler = x_scaler

        return x, y

    def loda_data(self):
        # 8:2
        x = self.x
        y = self.y

        # x, y = sklearn.utils.shuffle(x, y)

        # print(x.shape, y.shape)
        inx = int((len(x) * 0.8))
        x_train, x_test = x[:inx], x[inx:]
        y_train, y_test = y[:inx], y[inx:]
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)  # 打乱训练集
        return x_train, x_test, y_train, y_test

    def evaluates(self, y_true, y_pred):

        y_true = np.array(y_true).reshape((-1, 1))
        y_pred = np.array(y_pred).reshape((-1, 1))
        # print(y_true.shape)
        # print(y_pred.shape)

        # 未逆归一化
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        y_true_c = np.where(y_true >= 0, 1, 0)
        y_pred_c = np.where(y_pred >= 0, 1, 0)
        nacc = accuracy_score(y_true_c, y_pred_c)
        print('未逆归一化 mse:%f,mae:%f,nacc:%f' % (mse, mae, nacc))
        return mse, mae, nacc

    def write_log(self):
        now = datetime.datetime.now()
        now_time = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        with open("log_train.txt", "a", encoding="utf-8") as f:
            f.writelines([now_time, '\n'])
            f.writelines([self.ticker, ' ', self.net_model, ' ', self.loss_fun, '\n'])
            f.writelines(
                ['best_epoch:', str(self.best_epoch), ' best_mse:', str(self.best_mse), ' best_mae:',
                 str(self.best_mae),
                 ' best_nacc:', str(self.best_nacc), '\n'])
            f.writelines(['\n'])

    def train_main(self):
        # # 载入数据
        self.data_preprocessing(windows=50)
        x_train, x_test, y_train, y_test = self.loda_data()

        # 载入模型
        if self.net_model == 'lstm_1_net':
            model = lstm_1_net(shape=(50, self.k_long))
        elif self.net_model == 'lstm_2_net':
            model = lstm_2_net(shape=(50, self.k_long))
        elif self.net_model == 'lstm_3_net':
            model = lstm_3_net(shape=(50, self.k_long))
        elif self.net_model == 'gru_3_net':
            model = gru_3_net(shape=(50, self.k_long))
        elif self.net_model == 'rnn_3_net':
            model = rnn_3_net(shape=(50, self.k_long))
        elif self.net_model == 'bp_5_net':
            model = bp_5_net(shape=(50, self.k_long))
        elif self.net_model == 'nbeats':
            # time_steps, input_dim, output_dim = 50,  self.k_long, 1
            num_samples, time_steps, input_dim, output_dim = 50_000, 50, 1, 1
            # model = NBeatsKeras( backcast_length=time_steps, forecast_length=output_dim, share_weights_in_stack=True)
            # model.summary()
            model = NBeatsKeras(
                backcast_length=time_steps, forecast_length=output_dim,
                stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
                # nb_blocks_per_stack=2, thetas_dim=(4, 4),
                share_weights_in_stack=True,
                # hidden_layer_units=64
            )
        elif self.net_model == 'baselines':
            # Naive baselines
            print('Naive baselines')

            mse, mae, nacc = self.evaluates(y_test[:-1], y_test[1:])
            self.best_epoch = 'null'
            self.best_mse = mse
            self.best_mae = mae
            self.best_nacc = nacc
            print(self.best_epoch, self.best_mse, self.best_mae, self.best_nacc)
            self.write_log()
            return 0

        # loss选择
        if self.loss_fun == 'nloss':
            model.compile(loss=loss_nlossbc(a=1.1, b=0.2, c=0.1, d=0.9), optimizer='sgd',
                          metrics=['mse', 'mae', metrics()])
        elif self.loss_fun == 'mse':
            model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'mae', metrics()])
        elif self.loss_fun == 'mae':
            model.compile(loss='mae', optimizer='sgd', metrics=['mse', 'mae', metrics()])
        elif self.loss_fun == 'mbe':
            model.compile(loss=loss_mbe(), optimizer='sgd', metrics=['mse', 'mae', metrics()])

        # 训练模型
        his = model.fit(x_train, y_train, batch_size=64, epochs=self.epochs, validation_data=(x_test, y_test),
                        verbose=2)
        # 输出最优模型
        print(his.history)
        val = his.history['val_nacc']
        inx = max_index_max(val)
        self.best_epoch = inx
        self.best_mse = his.history['val_mse'][inx]
        self.best_mae = his.history['val_mae'][inx]
        self.best_nacc = his.history['val_nacc'][inx]
        print(self.best_epoch, self.best_mse, self.best_mae, self.best_nacc)
        self.write_log()


if __name__ == '__main__':
    # 设置随机性
    seed = 1998
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    model_list = ['baselines', 'nbeats', 'lstm_1_net', 'lstm_2_net', 'lstm_3_net', 'gru_3_net', 'rnn_3_net', 'bp_5_net']
    loss_list = ['nloss', 'mse', 'mae']
    ticker_list = ['000998', '600598']

    for k in ticker_list:
        for j in model_list:
            for i in loss_list:
                print(k, j, i)
                # ticker, loss_fun, net_model, epochs
                t = trainStockNet(k, i, j, 50)
                t.train_main()
