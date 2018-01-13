# Failed
import tensorflow as tf
import cfg
import sim_rnn_cfg
import numpy as np
import load_data
import data_set
import os
import sys
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = sim_rnn_cfg.visible_device


def my_rnn(cell, x, batch_size=sim_rnn_cfg.batch_size):
    batch_channels = tf.unstack(tf.transpose(x, (1, 0, 2)))
    state = cell.zero_state(batch_size, tf.float32)
    output = None
    for batch_channel in batch_channels:
        output, state = cell(batch_channel, state)
    return output


def my_bi_rnn(cell_f, cell_b, x, batch_size=sim_rnn_cfg.batch_size):
    batch_channels = tf.unstack(tf.transpose(x, (1, 0, 2)))
    state_f = cell_f.zero_state(batch_size, tf.float32)
    output_f = None
    state_b = cell_b.zero_state(batch_size, tf.float32)
    output_b = None
    for batch_channel in batch_channels:
        output_f, state_f = cell_f(batch_channel, state_f)
    for batch_channel_b in reversed(batch_channels):
        output_b, state_b = cell_b(batch_channel_b, state_b)
    if output_f is None or output_b is None:
        return None
    return tf.concat([output_f, output_b], axis=1)


# x shape [batch_size, seq]
def deep_nn(x, k_p):
    # print(x.shape)
    with tf.name_scope('reshape'):
        x_reshape = tf.reshape(x, [-1, sim_rnn_cfg.origin_d, 1])

    # cell = None
    # outputs = None
    if sim_rnn_cfg.rnn_cell_type.lower() == 'lstm':
        cell_func = tf.nn.rnn_cell.BasicLSTMCell
    elif sim_rnn_cfg.rnn_cell_type.lower() == 'block_lstm':
        cell_func = tf.contrib.rnn.LSTMBlockCell
    elif sim_rnn_cfg.rnn_cell_type.lower() == 'rnn':
        cell_func = tf.nn.rnn_cell.BasicRNNCell
    else:
        cell_func = tf.nn.rnn_cell.GRUCell
    # cell = cell_func(sim_rnn_cfg.rnn_num_units)
    # cell = tf.contrib.rnn.DropoutWrapper(cell)
    # print(x_reshape.shape)
    if sim_rnn_cfg.rnn_is_bidirection:
        cells_fw = tf.contrib.rnn.MultiRNNCell(
            [cell_func(sim_rnn_cfg.rnn_num_units) for _ in range(sim_rnn_cfg.rnn_layers)])
        cells_bw = tf.contrib.rnn.MultiRNNCell(
            [cell_func(sim_rnn_cfg.rnn_num_units) for _ in range(sim_rnn_cfg.rnn_layers)])
        last_output = my_bi_rnn(cells_fw, cells_bw, x_reshape)
        # last_output =
    else:
        cells = tf.contrib.rnn.MultiRNNCell(
            [cell_func(sim_rnn_cfg.rnn_num_units) for _ in range(sim_rnn_cfg.rnn_layers)])
        last_output = my_rnn(cells, x_reshape)

    # outputs [batch_size, rnn_num_units]
    with tf.name_scope('dropout'):
        rnn_out_drop = tf.nn.dropout(last_output, k_p)

    with tf.name_scope('dense'):
        y_rnn = tf.layers.dense(rnn_out_drop, len(cfg.classes))
    return y_rnn


def main(_):
    loop_epoch_num = sim_rnn_cfg.loop_epoch_num
    log_epoch_num = sim_rnn_cfg.log_epoch_num
    learning_rate = sim_rnn_cfg.learning_rate
    origin_d = sim_rnn_cfg.origin_d
    classes = cfg.classes
    t_v_t = load_data.TrainValiTest()
    t_v_t.load()
    train_samples, train_ls = t_v_t.train_samples_ls()
    vali_samples, vali_ls = t_v_t.vali_samples_ls()
    test_samples, test_ls = t_v_t.test_samples_ls()

    pca = PCA(sim_rnn_cfg.origin_d)
    pca.fit(train_samples, train_ls)
    train_samples = pca.transform(train_samples)
    vali_samples = pca.transform(vali_samples)
    test_samples = pca.transform(test_samples)

    train_set = data_set.DataSet(train_samples, train_ls)
    vali_set = data_set.DataSet(vali_samples, vali_ls)
    test_set = data_set.DataSet(test_samples, test_ls)

    x = tf.placeholder(tf.float32, [None, origin_d])
    y_ = tf.placeholder(tf.float32, [None, len(classes)])
    k_prob = tf.placeholder(tf.float32)

    y_rnn = deep_nn(x, k_prob)

    with tf.name_scope('loss'):
        cross_entroys = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_rnn)
    cross_entroy = tf.reduce_mean(cross_entroys)

    with tf.name_scope('adadelta_optimizer'):
        train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entroy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_rnn, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(loop_epoch_num):
            if i % log_epoch_num == 0:
                train_loss = loss_epoch(x, y_, k_prob, cross_entroy, train_set, sess)
                train_acc = acc_epoch(x, y_, k_prob, accuracy, train_set, sess)
                vali_loss = loss_epoch(x, y_, k_prob, cross_entroy, vali_set, sess)
                vali_acc = acc_epoch(x, y_, k_prob, accuracy, vali_set, sess)
                print('epoch %d , train_loss %g , train_acc %g '
                      ', vali_loss %g , vali_acc %g' % (i,
                                                        train_loss,
                                                        train_acc,
                                                        vali_loss,
                                                        vali_acc))
            train_epoch(x, y_, k_prob, train_step, train_set, sess)
            train_set.re_shuffle()
        test_loss = loss_epoch(x, y_, k_prob, cross_entroy, test_set, sess)
        test_acc = acc_epoch(x, y_, k_prob, accuracy, test_set, sess)
        print('test_loss %g , test_acc %g' % (test_loss, test_acc))


def train_epoch(x, y_, k_prob, train_step, train_set, sess):
    batch_size = sim_rnn_cfg.batch_size
    train_k_prob = sim_rnn_cfg.keep_prob
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = train_set.next_batch_fix2(batch_size)
        train_step.run(feed_dict={
            x: batch_x, y_: batch_y, k_prob: train_k_prob
        }, session=sess)


def loss_epoch(x, y_, k_prob, loss, d_set, sess):
    batch_size = sim_rnn_cfg.batch_size
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
    batch_size = sim_rnn_cfg.batch_size
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


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

