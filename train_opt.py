from train import trainStockNet
from Net import *
from StockLossAndMetrics import metrics, loss_nloss, loss_nlossbc, loss_mbe
import numpy as np
import random
import os
import tensorflow as tf
import optuna
import datetime
from nbeats_model import NBeatsNet as NBeatsKeras


def max_index_max(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return max(index)  # 返回一个列表


class trainStockNetOptuna(trainStockNet):
    def write_log(self):
        with open("log_opt.txt", "a", encoding="utf-8") as f:
            f.writelines(
                ['best_epoch:', str(self.best_epoch), ' best_mse:', str(self.best_mse), ' best_mae:',
                 str(self.best_mae),
                 ' best_nacc:', str(self.best_nacc), '\n'])

    def train_main(self):
        def train_main_fun(trial):
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
            a = trial.suggest_float('a', 1, 1.5)
            b = trial.suggest_float('b', 0, 0.5)
            c = trial.suggest_float('c', 0, 1)
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

            return self.best_nacc

        return train_main_fun


def opt_main(ticker, net_model, n_trials, epoch):
    now = datetime.datetime.now()
    now_time = str(now.strftime("%Y-%m-%d %H:%M:%S"))
    with open("log_opt.txt", "a", encoding="utf-8") as f:
        f.writelines([now_time, '\n'])
        f.writelines([ticker, ' ', net_model, '\n'])

    t = trainStockNetOptuna(ticker, 'nloss', net_model, epoch)
    t.data_preprocessing(windows=50)
    train_main_fun = t.train_main()

    # 贝叶斯优化
    study = optuna.create_study(direction='maximize', storage='sqlite:///db.sqlite3')
    study.optimize(train_main_fun, n_trials=n_trials)
    print(study.best_params, study.best_value)

    with open("log_opt.txt", "a", encoding="utf-8") as f:
        f.writelines(['best_params:', str(study.best_params), ' study.best_value:', str(study.best_value), '\n'])
        f.writelines(['best_params:', str(study.best_trial), '\n'])
        f.writelines(['\n'])


if __name__ == '__main__':
    # 设置随机性
    seed = 1998
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    model_list = ['nbeats', 'lstm_1_net', 'lstm_2_net', 'lstm_3_net', 'gru_3_net', 'rnn_3_net', 'bp_5_net']
    # model_list = ['gru_3_net', 'rnn_3_net', 'bp_5_net']
    # ticker_list = ['000998', '600598']
    ticker_list = ['002306']

    for k in ticker_list:
        for j in model_list:
            print(k, j, 'nloss_opt')
            opt_main(k, j, n_trials=75, epoch=35)
