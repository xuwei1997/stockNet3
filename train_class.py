import pandas as pd
import numpy as np
import random
import os
from sliding_window import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

class StockPrediction():
    def __init__(self, ticker):
        self.ticker = ticker

    def data_preprocessing(self, windows=50):
        # print('data_preprocessing')
        # kind 0-涨跌 1-收盘价
        # 取数据

        df1 = pd.read_excel('./data/d' + str(self.ticker) + '.xlsx')
        df2 = pd.read_csv('./data/m' + str(self.ticker) + '.csv')

        df = pd.merge(df1, df2, how='inner', on='tradeDate')
        df = df.fillna(0)

        key = ['preClosePrice', 'actPreClosePrice', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice',
               'turnoverVol', 'turnoverValue', 'dealAmount', 'turnoverRate', 'accumAdjFactor', 'negMarketValue',
               'marketValue', 'chgPct', 'PE_x', 'PE1', 'PB_x', 'vwap'] + df2.columns.tolist()[4:]
        key.pop(key.index('PE'))
        key.pop(key.index('PB'))

        self.k_long = len(key)

        x = np.array(df[key])
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x = x_scaler.fit_transform(x)
        print(x.shape)

        y = x[:, 5]
        pre_close = x[:, 0]


        # 滑动窗口
        x = sliding_window_view(x, (windows, self.k_long)).reshape((-1, windows, self.k_long))
        print('sliding_window_view')
        print(x.shape)
        print(y.shape)
        x = x[0:-1]
        y = y[windows:]
        pre_close = pre_close[windows:]
        print(x.shape)
        print(y.shape)
        print(pre_close.shape)

        self.x = x
        self.y = y
        self.pre_close = pre_close
        self.scaler = x_scaler

        return x, y, pre_close

    def loda_data(self):
        # 8:2
        x = self.x
        y = self.y
        pre_close = self.pre_close

        # x, y = sklearn.utils.shuffle(x, y)

        # print(x.shape, y.shape)
        inx = int((len(x) * 0.8))
        x_train, x_test = x[:inx], x[inx:]
        y_train, y_test = y[:inx], y[inx:]
        pre_close_train, pre_close_test = pre_close[:inx], pre_close[inx:]
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)  # 打乱训练集
        return x_train, x_test, y_train, y_test,pre_close_train, pre_close_test

    def evaluates(self, y_true0, y_pred0, y_pre_closs0):
        def inverse_transform_col(scaler, y, n_col):
            '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
            y = y.copy()
            y -= scaler.min_[n_col]
            y /= scaler.scale_[n_col]
            return y

        y_true0 = np.array(y_true0).reshape((-1, 1))
        y_pred0 = np.array(y_pred0).reshape((-1, 1))
        y_pre_closs0 = np.array(y_pre_closs0).reshape((-1, 1))


        y_true = y_true0 - y_pre_closs0
        y_pred = y_pred0 - y_pre_closs0
        # print(y_true.shape)
        # print(y_pred.shape)

        # 未逆归一化
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        y_true_c = np.where(y_true >= 0, 1, 0)
        y_pred_c = np.where(y_pred >= 0, 1, 0)
        nacc = accuracy_score(y_true_c, y_pred_c)
        print('未逆归一化 mse:%f,mae:%f,nacc:%f' % (mse, mae, nacc))

        # 逆归一化

        y_true1 = inverse_transform_col(self.scaler, y_true0, 5)
        y_pred1 = inverse_transform_col(self.scaler, y_pred0, 5)
        y_pre_closs1 = inverse_transform_col(self.scaler, y_pre_closs0, 0)

        y_true2 = y_true1 - y_pre_closs1
        y_pred2 = y_pred1 - y_pre_closs1

        # mse mae
        mse = mean_squared_error(y_true2, y_pred2)
        mae = mean_absolute_error(y_true2, y_pred2)

        # nacc
        y_true_c2 = np.where(y_true2 >= 0, 1, 0)
        y_pred_c2 = np.where(y_pred2 >= 0, 1, 0)
        # print(y_true_c==y_true_c2)
        nacc = accuracy_score(y_true_c2, y_pred_c2)

        print('逆归一化 mse:%f,mae:%f,nacc:%f' % (mse, mae, nacc))
