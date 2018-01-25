from keras import backend as K

def set_trainable(model, train):
    """
    Enable or disable training for the model
    args:
        model(?):
        train(?):
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train


def zero_loss(y_true, y_pred):
    """
    args:
        y_true():
        y_pred():
    """
    return K.zeros_like(y_true)

def sample_normal(args):
    """

    """
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return "%d sec" %s
    else:
        return "%d min %d sex" %(m, s)

