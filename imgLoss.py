import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from StockLossAndMetrics import loss_nlossbc
import tensorflow as tf


def loss_map(k):
    # print(k[0])
    a1 = tf.constant([k[0]])
    a2 = tf.constant([k[1]])
    # print(a1)

    nloss = loss_nlossbc(a=1.2, b=0.1, c=0.1, d=1 - 0.1)
    l = nloss(a1, a2)
    print(l)
    return l


def plt3D():
    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)

    X, Y = np.meshgrid(x, y)

    xs = X.flatten()
    ys = Y.flatten()
    print(xs.shape)
    print(ys.shape)

    zs = map(loss_map, zip(xs, ys))
    # #
    # a1 = tf.constant(xs)
    # a2 = tf.constant(ys)
    # nloss = loss_nlossbc()
    # zs = nloss(a1, a2)
    # print(zs)

    Z = np.array(list(zs)).reshape(X.shape)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    figure = plt.figure()
    ax = Axes3D(figure)

    surf = ax.plot_surface(X, Y, Z, rstride=1,  # rstride（row）指定行的跨度
                           cstride=1, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=5)
    ax.set_zlim(0, 8)
    figure.colorbar(surf, shrink=0.5, aspect=8)

    # plt.show()
    plt.savefig('loss_3d2.png', dpi=300)


if __name__ == '__main__':
    plt3D()
