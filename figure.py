from train import trainStockNet, max_index_max
from StockLossAndMetrics import metrics, loss_nloss, loss_nlossbc, loss_mbe
from Net import *
from nbeats_model import NBeatsNet as NBeatsKeras
import random
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


class printFig(trainStockNet):
    def __init__(self, ticker, net_model, opt_val, init_epochs):
        self.ticker = ticker
        self.net_model = net_model
        self.opt_val = opt_val
        self.epochs = init_epochs

    def inverse_transform_col(self, scaler, y, n_col):
        '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
        y = y.copy()
        y -= scaler.min_[n_col]
        y /= scaler.scale_[n_col]
        return y

    def fig_train(self, loss_fun):
        self.data_preprocessing(windows=50)
        x_train, x_test, y_train, y_test = self.loda_data()

        # print(x_test[:, 0, 0].shape)

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

        # loss选择
        if loss_fun == 'nloss':
            model.compile(loss=loss_nlossbc(a=1.1, b=0.2, c=0.1, d=0.9), optimizer='sgd',
                          metrics=['mse', 'mae', metrics()])
        elif loss_fun == 'mse':
            model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'mae', metrics()])
        elif loss_fun == 'mae':
            model.compile(loss='mae', optimizer='sgd', metrics=['mse', 'mae', metrics()])
        elif loss_fun == 'opt':
            a, b, c = self.opt_val
            model.compile(loss=loss_nlossbc(a=a, b=b, c=c, d=1 - c), optimizer='sgd', metrics=['mse', 'mae', metrics()])
        elif loss_fun == 'gt':
            # 返回收盘价
            preClosePrice0 = x_test[:, 0, 0]  # 归一化后的前收盘价
            preClosePrice = self.inverse_transform_col(self.scaler, preClosePrice0, 0)  # 归一化前的前收盘价
            y_pre = y_test.squeeze() * self.y_max  # 实际涨跌
            y_out = y_pre + preClosePrice

            return y_out[:100]

        # 设置每轮保存
        checkpoint = ModelCheckpoint('tmp/' + 'ep{epoch:d}.h5')

        # 训练模型
        his = model.fit(x_train, y_train, batch_size=64, epochs=self.epochs, validation_data=(x_test, y_test),
                        verbose=2, callbacks=[checkpoint])
        # 输出最优模型
        print(his.history)
        val = his.history['val_nacc']
        best_epoch = max_index_max(val) + 1
        print(best_epoch)

        # 载入最优模型并预测
        del model
        s = 'tmp/ep' + str(best_epoch) + '.h5'
        print(s)
        model2 = load_model(s, custom_objects={'nacc': metrics(), 'nlossbc': loss_nlossbc(a=1.1, b=0.2, c=0.1, d=0.9)})
        y_pre0 = model2.predict(x_test)
        y_pre0 = y_pre0.squeeze()


        # 返回收盘价
        preClosePrice0 = x_test[:, 0, 0]  # 归一化后的前收盘价
        # print(preClosePrice0.shape)
        preClosePrice = self.inverse_transform_col(self.scaler, preClosePrice0, 0)  # 归一化前的前收盘价
        # print(preClosePrice.shape)
        y_pre = y_pre0 * self.y_max  # 实际涨跌
        # print(y_pre0.shape)
        # print(self.y_max)
        # print(y_pre.shape)
        y_out = y_pre + preClosePrice
        print(y_out.shape)

        return y_out[:100]

    def fig_main(self):
        gt_p = self.fig_train('gt')
        mse_p = self.fig_train('mse')
        mae_p = self.fig_train('mae')
        nloss_p = self.fig_train('nloss')
        opt_p = self.fig_train('opt')

        gt_p = gt_p.squeeze()
        mse_p = mse_p.squeeze()
        mae_p = mae_p.squeeze()
        nloss_p = nloss_p.squeeze()
        opt_p = opt_p.squeeze()

        # gt_p=np.where(gt_p >= 0, 1, 0)
        # mse_p = np.where(mse_p >= 0, 1, 0)
        # mae_p= np.where(mae_p >= 0, 1, 0)
        # nloss_p = np.where(nloss_p >= 0, 1, 0)
        # opt_p = np.where(opt_p >= 0, 1, 0)

        # print(gt_p, mse_p, mae_p, nloss_p, opt_p)

        l1, = plt.plot(gt_p)
        l2, = plt.plot(mse_p)
        l3, = plt.plot(mae_p)
        l4, = plt.plot(nloss_p)
        l5, = plt.plot(opt_p)

        plt.legend()

        plt.legend(handles=[l1, l2, l3, l4, l5], labels=['GT', 'mse', 'mae', 'nloss', 'nloss_opt'])
        plt.title(self.ticker + ' ' + self.net_model)

        plt.show()


if __name__ == '__main__':
    # 设置随机性
    seed = 1998
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    #####设置参数#####
    ticker = '000998'
    # 'nbeats', 'lstm_1_net', 'lstm_2_net', 'lstm_3_net', 'gru_3_net', 'rnn_3_net', 'bp_5_net'
    net_model = 'nbeats'
    opt_val = [1.183940129252455, 0.016644080572874573, 0.562868861156576]  # 优化的参数abc
    #################

    print(ticker, net_model, opt_val)
    #  ticker, net_model, opt_val, init_epochs
    t = printFig(ticker, net_model, opt_val, init_epochs=50)
    t.fig_main()
