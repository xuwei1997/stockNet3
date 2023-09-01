from train_class import StockPrediction

import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from pylab import rcParams

# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.arima_model import ARIMA
# from pmdarima.arima import auto_arima
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from StockLossAndMetrics import metrics


class arimaPrediction(StockPrediction):
    def loda_data(self):
        # 8:2
        y = self.y
        pre_close = self.pre_close

        inx = int((len(y) * 0.8))
        train_data, test_data = y[:inx], y[inx:]
        pre_close_train, pre_close_test = pre_close[:inx], pre_close[inx:]

        return train_data, test_data, pre_close_train, pre_close_test

    def train_main(self):
        self.data_preprocessing(windows=50)
        train_data, test_data, pre_close_train, pre_close_test = self.loda_data()
        model = sm.tsa.arima.ARIMA(train_data, order=(3, 1, 2))
        fitted = model.fit()
        print(fitted.summary())

        # Forecast
        fc = fitted.forecast(len(test_data), alpha=0.05)

        # report performance
        mse = mean_squared_error(test_data, fc)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data, fc)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(fc - test_data) / np.abs(test_data))
        print('MAPE: ' + str(mape))
        # Around 3.5% MAPE implies the model is about 96.5% accurate in predicting the next 15 observations.

        # nacc
        nacc = self.evaluates(test_data, fc, pre_close_test)
        print('nacc: ' + str(nacc))


if __name__ == '__main__':
    ticker_list = ['000998', '600598']
    for k in ticker_list:
        print(k)
        t = arimaPrediction(ticker=k)
        t.train_main()
