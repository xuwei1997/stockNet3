# 基于考虑方向混合回归损失函数的神经网络股票涨跌额预测模型研究

## 摘要
基于神经网络的股票预测是当前经济学与计算机科学研究的重点，主要分为股票价格预测和股票涨跌预测。当前，价格预测模型构建时通常仅度量预测值和真实值的差，忽略了度量涨跌方向准确性；涨跌预测仅关注涨跌方向，丢失了大量关于股价的信息。为此，本文提出一种考虑方向的混合回归损失函数(Hybridised Loss Functions of Considering Direction, CDHloss)，在传统均方误差损失上针对符号不一致项添加惩罚，并融入了分类交叉熵损失，该损失函数可与各类神经网络组合为股票涨跌额预测模型。进一步的，以均方误差（Mean Square Error， MSE）和方向准确率（Direction Accuracy， DA）为目标，采用多目标贝叶斯优化方法获取损失函数中系数的帕累托最优前沿。本文从理论角度论证了损失函数性能的优越性，并选取隆平高科、恒瑞医药、中科云网3050日的交易数据及量化因子进行实验。结果表明，考虑方向的混合回归损失函数在以BP、LSTM、RNN、GRU与BEATS等网络为主干的预测模型中，方向准确率与均方误差损失函数和绝对值误差损失函数相比有2%-9%的提升，且均方误差评价指标达到相同水平。与Baseline、 ARIMA、高斯过程、RF-LSTM模型相比，结合损失函数的方向准确率提升3%-14%。本文提出的损失函数驱动神经网络模型在准确预测股价的同时提供准确的涨跌方向，为投资者提供了更加准确的决策信息。

## 文件结构
1. StockLossAndMetrics.py 定义CDHloss和评价指标
2. Net.py 构建训练中使用的模型
2. train.py 训练模型的mse,mae,CDHloss，结果导出为log_train.txt
2. train_opt.py 对CDHloss进行单目标(nacc)贝叶斯优化，结果导出为log_opt.txt
3. train_MOBOpt.py 对CDHloss进行多目标贝叶斯优化，结果导出为log_opt.txt
4. ARIMA.py ARIMA算法
5. Gaussian_Process.py 高斯过程
6. RF-LSTM.py RF-LSTM算法（决策树）
7. GA-LSTM.py GA-LSTM算法，不稳定，不建议使用
8. figure.py 画图，单个股票单个模型在测试集最后100个股价上的拟合情况
9. imgLoss.py 画CDHLoss示意图
9. train_temp.py 将CDHloss拓展到温度预测上
10. train_MOBOpt_temp.py 将对CDHloss的多目标贝叶斯优化拓展到温度预测上