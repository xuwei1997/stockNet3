from train_class import StockPrediction
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class gaussianPrediction(StockPrediction):
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

        return x_train, x_test, y_train, y_test, pre_close_train, pre_close_test

    def train_main(self):
        self.data_preprocessing(windows=50)
        x_train, x_test, y_train, y_test, pre_close_train, pre_close_test = self.loda_data()
        gpr = GaussianProcessRegressor()
        gpr.fit(x_train, y_train)
        y_pre = gpr.predict(x_test)
        nacc = self.evaluates(y_test, y_pre, pre_close_test)
        # print('nacc: ' + str(nacc))


if __name__ == '__main__':
    ticker_list = ['002186', '000157']
    for k in ticker_list:
        print(k)
        t = gaussianPrediction(ticker=k)
        t.train_main()
