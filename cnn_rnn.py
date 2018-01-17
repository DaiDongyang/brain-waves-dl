import tensorflow as tf
import cfg
import numpy as np
import load_data
import data_set
import os
import sys
import time

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
        with tf.name_scope('reshape_rnn'):
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
    loop_epoch_num = cfg.loop_epoch_num
    log_epoch_num = cfg.log_epoch_num
    loss_weights = tf.constant(cfg.loss_weights, dtype=tf.float32)
    learning_rate = cfg.learning_rate
    origin_d = cfg.origin_d
    classes = cfg.classes
    fc_layers = len(cfg.fc_w_shapes)
    t_v_t = load_data.TrainValiTest()
    t_v_t.load()
    train_samples, train_ls = t_v_t.train_samples_ls()
    vali_samples, vali_ls = t_v_t.vali_samples_ls()
    test_samples, test_ls = t_v_t.test_samples_ls()

    # pca = PCA(sim_rnn_cfg.origin_d)
    # pca.fit(train_samples, train_ls)
    # train_samples = pca.transform(train_samples)
    # vali_samples = pca.transform(vali_samples)
    # test_samples = pca.transform(test_samples)

    train_set = data_set.DataSet(train_samples, train_ls)
    vali_set = data_set.DataSet(vali_samples, vali_ls)
    test_set = data_set.DataSet(test_samples, test_ls)

    x = tf.placeholder(tf.float32, [None, origin_d])
    y_ = tf.placeholder(tf.float32, [None, len(classes)])
    k_prob = tf.placeholder(tf.float32, [fc_layers + 1])

    y_nn = deep_nn(x, k_prob)

    with tf.name_scope('loss'):
        weights = tf.reduce_sum(loss_weights * y_, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn)
        weight_losses = unweighted_losses * weights
    cross_entroy = tf.reduce_mean(weight_losses)

    with tf.name_scope('adadelta_optimizer'):
        train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entroy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    start = time.time()
    with tf.Session(config=config) as sess:
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
    end = time.time()
    print("time ", (end - start) / 3600)


def train_epoch(x, y_, k_prob, train_step, train_set, sess):
    batch_size = cfg.batch_size
    train_k_prob = cfg.dropout_probs
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = train_set.next_batch_fix2(batch_size)
        train_step.run(feed_dict={
            x: batch_x, y_: batch_y, k_prob: train_k_prob
        }, session=sess)


def loss_epoch(x, y_, k_prob, loss, d_set, sess):
    batch_size = cfg.batch_size
    dropout_probs_size = len(cfg.dropout_probs)
    losses = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_loss = loss.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: np.ones(dropout_probs_size)
        }, session=sess)
        losses.append(batch_loss)
        weights.append(len(batch_x))
    return float(np.dot(losses, weights) / np.sum(weights))


def acc_epoch(x, y_, k_prob, acc, d_set, sess):
    batch_size = cfg.batch_size
    dropout_probs_size = len(cfg.dropout_probs)
    acces = list()
    weights = list()
    is_epoch_end = False
    while not is_epoch_end:
        batch_x, batch_y, is_epoch_end = d_set.next_batch_fix2(batch_size)
        batch_acc = acc.eval(feed_dict={
            x: batch_x, y_: batch_y, k_prob: np.ones(dropout_probs_size)
        }, session=sess)
        acces.append(batch_acc)
        weights.append(len(batch_x))
    return float(np.dot(acces, weights) / np.sum(weights))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
