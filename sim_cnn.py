import tensorflow as tf
import sim_cnn_cfg


def deep_nn(x, k_prob):

    with tf.name_scope('reshape'):
        x_reshape = tf.reshape(x, [-1, sim_cnn_cfg.origin_d, 1])

    with tf.name_scope('conv0'):
        w_conv0 = weight_variable(sim_cnn_cfg.conv_ws[0])
        b_conv0 = bias_variable(sim_cnn_cfg.conv_ws[0][-1])
        h_conv0 = tf.nn.relu(conv1d(x_reshape, w_conv0) + b_conv0)

    with tf.name_scope('pool0'):
        h_pool0 = pool1d(h_conv0)

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable(sim_cnn_cfg.conv_ws[1])
        b_conv1 = bias_variable(sim_cnn_cfg.conv_ws[1][-1])
        h_conv1 = tf.nn.relu(conv1d(h_pool0, w_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = pool1d(h_conv1)

    with tf.name_scope('fc0'):
        w_fc0 = weight_variable((sim_cnn_cfg.fc_ds[0], sim_cnn_cfg.fc_ds[1]))
        b_fc0 = bias_variable(sim_cnn_cfg.fc_ds[1])
        d1, d2, d3 = h_pool1.shape()
        h_pool1_flat = tf.reshape((d1, d2*d3))
        h_fc0 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc0) + b_fc0)

    with tf.name_scope('dropout'):
        h_fc0_drop = tf.nn.dropout(h_fc0, k_prob)

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable((sim_cnn_cfg.fc_ds[1], sim_cnn_cfg.fc_ds[2]))
        b_fc1 = bias_variable(sim_cnn_cfg.fc_ds[2])
        y_fc1 = tf.matmul(h_fc0_drop, w_fc1) + b_fc1

    return y_fc1


# x, W is 3D tensor
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=sim_cnn_cfg.conv_stride, padding=sim_cnn_cfg.conv_padding)


# x is 3D tensor
def pool1d(x):
    if sim_cnn_cfg.pool_type == 0:
        return tf.nn.avg_pool(x, ksize=sim_cnn_cfg.pool_ksize, strides=sim_cnn_cfg.pool_strides,
                              padding=sim_cnn_cfg.pool_padding)
    else:
        return tf.nn.max_pool(x, ksize=sim_cnn_cfg.pool_ksize, strides=sim_cnn_cfg.pool_strides,
                              padding=sim_cnn_cfg.pool_padding)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
