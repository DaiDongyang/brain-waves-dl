import tensorflow as tf
import cfg
import numpy as np
import load_data
import data_set
import os
import sys
import time
import operator
import post_process
from itertools import accumulate
from functools import reduce

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_device


# x : 2d tensor
# k_probs, list of placeholder p for dropout, two elements
def deep_nn(x, k_probs):
    # k_probs_list = k_probs.eval()
    with tf.name_scope('reshape0'):
        x_3d = tf.reshape(x, [-1, cfg.origin_d, 1])

    with tf.name_scope('cnn'):
        h_cnn = cnn(x_3d)

    # with tf.name_scope('dropout0'):
    #     h_cnn_drop = tf.nn.dropout(h_cnn, k_probs[0])
    with tf.name_scope('rnn'):
        h_rnn = rnn(h_cnn)

    with tf.name_scope('dropout0'):
        h_rnn_drop = tf.nn.dropout(h_rnn, k_probs[0])

    with tf.name_scope('fc'):
        y_nn = fc(h_rnn_drop, k_probs, 1)

    return y_nn


# x is a 3d tensor, [batch_num, features, channels]
def cnn(x, cnn_fs=cfg.conv_fs):
    for cnn_f in cnn_fs:
        w = weight_variable(cnn_f)
        b = bias_variable(cnn_f[-1:])
        h = tf.nn.relu(conv1d(x, w) + b)
        x = pool1d(h)
    return x


def rnn(x, rnn_units_ls=cfg.rnn_units_list, cell_type=cfg.rnn_cell_type,
        is_bi=cfg.rnn_is_bidirection):
    if len(rnn_units_ls) < 1:
        with tf.name_scope('reshape1'):
            [d1, d2, d3] = x.shape.as_list()
            x_flat = tf.reshape(x, [-1, d2 * d3])
            return x_flat
    if cell_type.lower() == 'lstm':
        cell_func = tf.nn.rnn_cell.BasicLSTMCell
    elif cell_type.lower() == 'block_lstm':
        cell_func = tf.contrib.rnn.LSTMBlockCell
    elif cell_type.lower() == 'rnn':
        cell_func = tf.nn.rnn_cell.BasicRNNCell
    else:
        cell_func = tf.nn.rnn_cell.GRUCell

    if is_bi:
        cells_fw = tf.contrib.rnn.MultiRNNCell([cell_func(rnn_unit) for rnn_unit in rnn_units_ls])
        cells_bw = tf.contrib.rnn.MultiRNNCell([cell_func(rnn_unit) for rnn_unit in rnn_units_ls])
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, x,
                                                         dtype=tf.float32, scope='rnn_fbw')
        fw_outputs = rnn_outputs[0][:, -1, :]
        bw_outputs = rnn_outputs[1][:, -1, :]
        last_output = tf.concat([fw_outputs, bw_outputs], axis=1)
    else:
        cells = tf.contrib.rnn.MultiRNNCell([cell_func(rnn_unit) for rnn_unit in rnn_units_ls])
        rnn_outputs, _ = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32, scope='rnn_fw')
        last_output = rnn_outputs[:, -1, :]
    return last_output


def fc(x, k_probs, start_prob_ind, w_shapes=cfg.fc_w_shapes):
    for i, w_shape in enumerate(w_shapes):
        w_fc = weight_variable(w_shape)
        b_fc = bias_variable(w_shape[-1:])
        k_prob = k_probs[start_prob_ind + i]
        if w_shape is w_shape[-1]:
            h = tf.matmul(x, w_fc) + b_fc
        else:
            h = tf.nn.relu(tf.matmul(x, w_fc) + b_fc)
        x = tf.nn.dropout(h, k_prob)
    return x


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# x, W is 3D tensor
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=cfg.conv_stride, padding=cfg.conv_padding)


# x is 3D tensor
def pool1d(x):
    if cfg.cnn_pool_type == 0:
        pooling_type = 'AVG'
    else:
        pooling_type = 'MAX'
    return tf.nn.pool(x, cfg.cnn_pool_ksize, pooling_type=pooling_type,
                      padding=cfg.cnn_pool_padding,
                      strides=cfg.cnn_pool_strides)


def main(_):
    log_epoch_num = cfg.log_epoch_num
    loop_epoch_nums = cfg.loop_epoch_nums
    learning_rates = cfg.learning_rates
    loss_weight = tf.constant(cfg.loss_weights, dtype=tf.float32)
    optimizer_type = cfg.optimizer_type
    batch_size = cfg.batch_size

    origin_d = cfg.origin_d
    n_classes = len(cfg.classes)
    probs_size = len(cfg.dropout_probs)

    restore_file = cfg.restore_file
    restart_epoch_i = cfg.restart_epoch_i
    persist_checkpoint_interval = cfg.persist_checkpoint_interval
    persist_checkpoint_file = cfg.persist_checkpoint_file

    tvt = load_data.TrainValiTest()
    tvt.load()
    train_samples, train_ls = tvt.train_samples_ls()
    vali_samples, vali_ls = tvt.vali_samples_ls()
    test_samples, test_ls = tvt.test_samples_ls()

    train_set = data_set.DataSet(train_samples, train_ls)
    vali_set = data_set.DataSet(vali_samples, vali_ls)
    test_set = data_set.DataSet(test_samples, test_ls)

    x = tf.placeholder(tf.float32, [None, origin_d])
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    k_probs_ph = tf.placeholder(tf.float32, [probs_size])
    lr_ph = tf.placeholder(tf.float32)

    y_nn = deep_nn(x, k_probs_ph)

    with tf.name_scope('loss'):
        weights = tf.reduce_sum(loss_weight * y_, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn)
        weight_losses = unweighted_losses * weights
    loss = tf.reduce_mean(weight_losses)

    with tf.name_scope('optimizer'):
        if optimizer_type.lower() == 'adam':
            train_step = tf.train.AdamOptimizer(lr_ph).minimize(loss)
        elif optimizer_type.lower() == 'adadelta':
            train_step = tf.train.AdadeltaOptimizer(lr_ph).minimize(loss)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr_ph).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    # sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    with tf.Session(config=sess_config) as sess:
        # merged_summary_op = tf.merge_all_summaries()
        tf.summary.scalar("loss", loss)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        start = time.time()
        if cfg.is_train:
            start_i = 0
            end_i = reduce((lambda _a, _b: _a + _b), loop_epoch_nums, 0)
            if cfg.is_restore:
                saver.restore(sess, restore_file)
                start_i = restart_epoch_i
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            for i in range(start_i, end_i):
                if i % log_epoch_num == 0:
                    train_acc, train_loss = acc_loss_epoch(x, y_, k_probs_ph, accuracy, loss,
                                                           train_set, sess)
                    vali_acc, vali_loss = acc_loss_epoch(x, y_, k_probs_ph, accuracy, loss,
                                                         vali_set, sess)
                    print('epoch %d , train_acc %g , train_loss %g , vali_acc %g , vali_loss %g' % (
                        i, train_acc, train_loss, vali_acc, vali_loss))
                if i % persist_checkpoint_interval == 0 and i >= persist_checkpoint_interval:
                    saver.save(sess, persist_checkpoint_file+str(i))
                lr = get_lr(learning_rates, loop_epoch_nums, i)
                train_epoch(x, y_, k_probs_ph, train_step, lr_ph, lr, train_set, sess)
                # summary_str = sess.run(merged_summary_op, feed_dict={
                #
                # })
                # summary_writer.add_summary(summary_str, train_set.batch_num(batch_size) * i)
        else:
            saver.restore(sess, restore_file)
        test_acc, test_loss = acc_loss_epoch(x, y_, k_probs_ph, accuracy, loss, test_set, sess)
        print('test_acc %g , test_loss %g' % (test_acc, test_loss))
        end = time.time()
        print('total time %g s' % (end-start))
        save_result(x, k_probs_ph, y_nn, test_set, sess)


def save_result(x, k_probs_ph, y_nn, d_set, sess):
    batch_size = cfg.batch_size
    gt_pickle = cfg.gt_pickle
    pr_pickle = cfg.pr_pickle
    dropout_probs_size = len(cfg.dropout_probs)
    g_ts = list()
    p_rs = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_y_nn = y_nn.eval(feed_dict={
            x: batch_x,
            k_probs_ph: np.ones(dropout_probs_size)
        }, session=sess)
        batch_p_r = np.argmax(batch_y_nn, 1)
        batch_g_t = np.argmax(batch_y, 1)
        g_ts += list(batch_g_t)
        p_rs += list(batch_p_r)
    post_process.dump_list(g_ts, gt_pickle)
    post_process.dump_list(p_rs, pr_pickle)


def train_epoch(x, y_, k_probs_ph, train_step, lr_ph, lr, train_set, sess):
    batch_size = cfg.batch_size
    train_k_probs = cfg.dropout_probs
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = train_set.next_batch_fix2(batch_size)
        train_step.run(feed_dict={
            x: batch_x,
            y_: batch_y,
            k_probs_ph: train_k_probs,
            lr_ph: lr
        }, session=sess)


def acc_loss_epoch(x, y_, k_probs_ph, acc_tf, loss_tf, d_set, sess):
    batch_size = cfg.batch_size
    dropout_probs_size = len(cfg.dropout_probs)
    acces = list()
    losses = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_acc = acc_tf.eval(feed_dict={
            x: batch_x,
            y_: batch_y,
            k_probs_ph: np.ones(dropout_probs_size)
        }, session=sess)
        batch_loss = loss_tf.eval(feed_dict={
            x: batch_x,
            y_: batch_y,
            k_probs_ph: np.ones(dropout_probs_size)
        }, session=sess)
        acces.append(batch_acc)
        losses.append(batch_loss)
        weights.append(len(batch_x))
    epoch_acc = float(np.dot(acces, weights) / np.sum(weights))
    epoch_loss = float(np.dot(losses, weights) / np.sum(weights))
    return epoch_acc, epoch_loss


def get_lr(lrs, train_epoch_nums, current_num):
    acc_train_epoch_nums = accumulate(train_epoch_nums, operator.add)
    for lr, acc_train_epoch_num in zip(lrs, acc_train_epoch_nums):
        if current_num < acc_train_epoch_num:
            return lr
    return lrs[-1]


def print_config(cfg_file='./cfg.py'):
    with open(cfg_file, 'r') as cfg_f:
        for line in cfg_f:
            if '#' in line:
                if 'old' in line:
                    break
                continue
            if '=' in line:
                print(line)


if __name__ == '__main__':
    print_config()
    tf.app.run(main=main, argv=[sys.argv[0]])
    print('id_str', cfg.id_str)

