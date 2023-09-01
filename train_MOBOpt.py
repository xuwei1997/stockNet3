# 在不同的模型上运用CDHloss损失函数预测股票涨跌额，并采用MOBOpt贝叶斯优化寻找CDHloss最佳的值
from train import trainStockNet
from Net import *
from StockLossAndMetrics import metrics, loss_nloss, loss_nlossbc, loss_mbe
import numpy as np
import random
import os
import tensorflow as tf
import datetime
from nbeats_model import NBeatsNet as NBeatsKeras
import mobopt as mo


def max_index_max(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return max(index)  # 返回一个列表


class trainMOBOpt(trainStockNet):
    def __init__(self, ticker, loss_fun, net_model, epochs):
        super().__init__(ticker, loss_fun, net_model, epochs)
        self.x_opt = None

    def write_log(self):
        with open("log_opt.txt", "a", encoding="utf-8") as f:
            f.writelines(
                ['best_epoch:', str(self.x_opt), ' best_epoch:', str(self.best_epoch), ' best_mse:', str(self.best_mse),
                 ' best_mae:',
                 str(self.best_mae),
                 ' best_nacc:', str(self.best_nacc), '\n'])

    def train_main(self):
        def train_main_fun(x):
            try:
                self.x_opt = x
                print(x)
                # # 载入数据

                x_train, x_test, y_train, y_test = self.loda_data()
                print(x_train.shape)
                print(y_train.shape)

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

                # 编译模型
                a = x[0]
                b = 0
                c = x[1]

                model.compile(loss=loss_nlossbc(a=a, b=b, c=c, d=(1 - c)), optimizer='sgd',
                              metrics=['mse', 'mae', metrics()])

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
                self.write_log()

                # nan
                if np.isnan(self.best_nacc) or np.isnan(self.best_mse):
                    print('Nan')
                    self.best_nacc = 0.4
                    self.best_mse = 1

                # 多目标优化，最大值
                print([self.best_nacc, -self.best_mse * 10])
                return [self.best_nacc, -self.best_mse * 10]

            except BaseException as e:
                print('except:', e)
                self.best_nacc = 0.4
                self.best_mse = 1
                return [self.best_nacc, -self.best_mse * 10]

        return train_main_fun


def opt_main(ticker, net_model, n_trials, epoch):
    now = datetime.datetime.now()
    now_time = str(now.strftime("%Y-%m-%d %H:%M:%S"))
    with open("log_opt.txt", "a", encoding="utf-8") as f:
        f.writelines([now_time, '\n'])
        f.writelines([ticker, ' ', net_model, '\n'])

    # 初始化
    t = trainMOBOpt(ticker, 'nloss', net_model, epoch)
    t.data_preprocessing(windows=50)
    # 优化目标函数 a = x[0] c = x[1]
    train_main_fun = t.train_main()

    # 贝叶斯优化
    pbounds = np.array([[1, 1.5], [0, 1]])
    Optimize = mo.MOBayesianOpt(target=train_main_fun, NObj=2, pbounds=pbounds, constraints=[],
                                verbose=False, Picture=False, TPF=None,
                                n_restarts_optimizer=10, Filename='optf.txt', MetricsPS=True,
                                max_or_min='max', RandomSeed=42)
    Optimize.initialize(init_points=5)
    front, pop = Optimize.maximize(n_iter=n_trials)

    print(front)
    print(pop)
    print(Optimize.x_Pareto)
    print(Optimize.y_Pareto)

    with open("log_opt.txt", "a", encoding="utf-8") as f:
        f.writelines(['front:', str(front), '\n'])
        f.writelines(['pop:', str(pop), '\n'])
        f.writelines(['x_Pareto:', str(Optimize.x_Pareto), '\n'])
        f.writelines(['y_Pareto:', str(Optimize.y_Pareto), '\n'])
        f.writelines(['\n'])


if __name__ == '__main__':
    # 设置随机性
    seed = 1998
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    model_list = ['lstm_3_net', 'lstm_2_net', 'lstm_1_net', 'gru_3_net', 'nbeats', 'rnn_3_net', 'bp_5_net']
    # model_list = ['bp_5_net']
    ticker_list = ['600276', '002306']
    # ticker_list = ['000998']

    for k in ticker_list:
        for j in model_list:
            print(k, j, 'nloss_opt')
            opt_main(k, j, n_trials=75, epoch=35)
            # opt_main(k, j, n_trials=5, epoch=5)
