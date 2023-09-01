import tensorflow as tf


def conversion(y):  # 大于等于0的置1,小于0的置0
    # print(y)
    # 添加小数字
    y = y + 1e-5
    # 大于等于0的置1,小于0的置0
    y = tf.where(tf.greater_equal(y, 0), tf.math.floordiv(y, y), tf.math.subtract(y, y))
    # y = tf.where(tf.greater_equal(y, 0), 1, 0)
    # print(y)
    return y


def metrics():
    def nacc(y_true, y_pred):
        # print(y_true)
        y_true = conversion(y_true)
        y_pred = conversion(y_pred)
        acc = tf.keras.metrics.binary_accuracy(y_true, y_pred)
        return acc

    return nacc


def loss_nloss(a=2, b=0.5):
    def nloss(y_true, y_pred):
        key_true = tf.greater_equal(y_true, 0)
        key_pred = tf.greater_equal(y_pred, 0)
        key = key_true ^ key_pred
        y_pred = tf.where(key, a * y_pred + b, y_pred)
        nlossacc = tf.keras.losses.mean_squared_error(y_true, y_pred)
        # nlossacc = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        return nlossacc

    return nloss


def loss_nlossbc(a=1, b=0, c=0.5, d=0.5):
    def nlossbc(y_true, y_pred):
        key_true = tf.greater_equal(y_true, 0)
        key_pred = tf.greater_equal(y_pred, 0)
        key = key_true ^ key_pred
        y_pred = tf.where(key, a * y_pred + b, y_pred)
        nloss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        class_true = conversion(y_true)
        class_pred = conversion(y_pred)
        nacc = tf.keras.losses.binary_crossentropy(class_true, class_pred)

        l = c * nacc + d * nloss

        return l

    return nlossbc


def loss_mbe():
    def mbe(y_true, y_pred):
        diff = tf.math.subtract(y_true , y_pred)
        l = tf.math.reduce_mean(diff)
        return l

    return mbe
