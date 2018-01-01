import tensorflow as tf
import numpy as np
import tempfile
import sys
import os
import sim_cnn_cfg
import cfg
import load_data
import data_set

os.environ["CUDA_VISIBLE_DEVICES"] = sim_cnn_cfg.visible_device


def deep_nn(x, k_prob):
    with tf.name_scope('reshape'):
        x_reshape = tf.reshape(x, [-1, sim_cnn_cfg.origin_d, 1])

    with tf.name_scope('conv0'):
        w_conv0 = weight_variable(sim_cnn_cfg.conv_ws[0])
        b_conv0 = bias_variable([sim_cnn_cfg.conv_ws[0][-1]])
        h_conv0 = tf.nn.relu(conv1d(x_reshape, w_conv0) + b_conv0)

    with tf.name_scope('pool0'):
        h_pool0 = pool1d(h_conv0)

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable(sim_cnn_cfg.conv_ws[1])
        b_conv1 = bias_variable([sim_cnn_cfg.conv_ws[1][-1]])
        h_conv1 = tf.nn.relu(conv1d(h_pool0, w_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = pool1d(h_conv1)

    with tf.name_scope('fc0'):
        w_fc0 = weight_variable((sim_cnn_cfg.fc_ds[0], sim_cnn_cfg.fc_ds[1]))
        b_fc0 = bias_variable([sim_cnn_cfg.fc_ds[1]])
        [d1, d2, d3] = h_pool1.shape.as_list()
        h_pool1_flat = tf.reshape(h_pool1, [-1, d2 * d3])
        h_fc0 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc0) + b_fc0)

    with tf.name_scope('dropout'):
        h_fc0_drop = tf.nn.dropout(h_fc0, k_prob)

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable((sim_cnn_cfg.fc_ds[1], sim_cnn_cfg.fc_ds[2]))
        b_fc1 = bias_variable([sim_cnn_cfg.fc_ds[2]])
        y_fc1 = tf.matmul(h_fc0_drop, w_fc1) + b_fc1

    return y_fc1


# x, W is 3D tensor
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=sim_cnn_cfg.conv_stride, padding=sim_cnn_cfg.conv_padding)


# x is 3D tensor
def pool1d(x):
    if sim_cnn_cfg.pool_type == 0:
        pooling_type = 'AVG'
    else:
        pooling_type = 'MAX'
    return tf.nn.pool(x, sim_cnn_cfg.pool_ksize, pooling_type=pooling_type, padding=sim_cnn_cfg.pool_padding,
                      strides=sim_cnn_cfg.pool_strides)


# def pool1d(x):
#     if sim_cnn_cfg.pool_type == 0:
#         return tf.nn.avg_pool(x, ksize=sim_cnn_cfg.pool_ksize, strides=sim_cnn_cfg.pool_strides,
#                               padding=sim_cnn_cfg.pool_padding)
#     else:
#         return tf.nn.max_pool(x, ksize=sim_cnn_cfg.pool_ksize, strides=sim_cnn_cfg.pool_strides,
#                               padding=sim_cnn_cfg.pool_padding)


def main(_):
    # load config here
    batch_size = sim_cnn_cfg.batch_size
    loop_epoch_num = sim_cnn_cfg.loop_epoch_num
    log_epoch_num = sim_cnn_cfg.log_epoch_num
    persist_epoch_num = sim_cnn_cfg.persist_epoch_num
    save_epoch_num = sim_cnn_cfg.save_epoch_num
    learning_rate = sim_cnn_cfg.learning_rate
    origin_d = sim_cnn_cfg.origin_d
    # train_k_prob = sim_cnn_cfg.keep_prob
    classes = cfg.classes

    t_v_t = load_data.TrainValiTest()
    t_v_t.load()
    train_samples, train_ls = t_v_t.train_samples_ls()
    vali_samples, vali_ls = t_v_t.vali_samples_ls()
    test_samples, test_ls = t_v_t.test_samples_ls()

    train_set = data_set.DataSet(train_samples, train_ls)
    vali_set = data_set.DataSet(vali_samples, vali_ls)
    test_set = data_set.DataSet(test_samples, test_ls)

    x = tf.placeholder(tf.float32, [None, origin_d])
    y_ = tf.placeholder(tf.float32, [None, len(classes)])
    k_prob = tf.placeholder(tf.float32)

    y_conv = deep_nn(x, k_prob)

    with tf.name_scope('loss'):
        cross_entroys = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entroy = tf.reduce_mean(cross_entroys)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entroy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('result'):
        g_t = tf.argmax(y_, 1)
        p_r = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(loop_epoch_num):
            train_epoch(x, y_, k_prob, train_step, train_set, sess)
            train_set.re_shuffle()
            if i % log_epoch_num == 0:
                train_loss = loss_epoch(x, y_, k_prob, cross_entroy, train_set, sess)
                train_acc = acc_epoch(x, y_, k_prob, accuracy, train_set, sess)
                vali_loss = loss_epoch(x, y_, k_prob, cross_entroy, vali_set, sess)
                vali_acc = acc_epoch(x, y_, k_prob, accuracy, vali_set, sess)
                print('epoch %d , train_loss %g , train_acc %g , vali_loss %g , vali_acc %g' % (i,
                                                                                                train_loss, train_acc,
                                                                                                vali_loss, vali_acc))
        test_loss = loss_epoch(x, y_, k_prob, cross_entroy, test_set, sess)
        test_acc = acc_epoch(x, y_, k_prob, accuracy, test_set, sess)
        print('test_loss %g , test_acc %g' % (test_loss, test_acc))


def train_epoch(x, y_, k_prob, train_step, train_set, sess):
    batch_size = sim_cnn_cfg.batch_size
    train_k_prob = sim_cnn_cfg.keep_prob
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = train_set.next_batch_fix2(batch_size)
        train_step.run(feed_dict={
            x: batch_x, y_: batch_y, k_prob: train_k_prob
        }, session=sess)


def loss_epoch(x, y_, k_prob, loss, d_set, sess):
    batch_size = sim_cnn_cfg.batch_size
    losses = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_loss = loss.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: 1
        }, session=sess)
        losses.append(batch_loss)
        weights.append(len(batch_x))
    return float(np.dot(losses, weights) / np.sum(weights))


def acc_epoch(x, y_, k_prob, acc, d_set, sess):
    batch_size = sim_cnn_cfg.batch_size
    acces = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_acc = acc.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: 1
        }, session=sess)
        acces.append(batch_acc)
        weights.append(len(batch_x))
    return float(np.dot(acces, weights) / np.sum(weights))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
