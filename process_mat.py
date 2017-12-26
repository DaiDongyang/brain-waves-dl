import scipy.io as sio
import numpy as np
import os
from collections import Counter

origin_dim = 1000


def save_train_mat():
    preffix = './data'
    mat_name = './train_1222_578001438.mat'
    d = sio.loadmat(mat_name)
    for k, v in d.items():
        if type(v) is np.ndarray:
            file_path = os.path.join(preffix, k + '.npy')
            print(file_path)
            np.save(file_path, v)


def load_pair(mat_file_path, label_file_path):
    """
    load data from a mat file and its corresponding file, for example,
    this function can be called like "load_pair('./mats/sleep_data_row3_1.mat','./labels/HypnogramAASM_subject1.txt')"
    :param mat_file_path: the path of mat file
    :param label_file_path: the path of label(txt) file
    :return:
        instances: a list of features (list of list, can be converted into numpy matrix)
        labels: a list of labels, corresponding to list of features
    """
    mat = sio.loadmat(mat_file_path)
    instances = list(mat['data'].reshape((-1, origin_dim)))
    labels = list()
    with open(label_file_path, 'r') as label_f:
        next(label_f)
        for line in label_f:
            label = int(line.strip())
            labels.append(label)
    return instances, labels


def load_pair_np(mat_file_path, label_file_path):
    """load a pair and return np"""
    instances, labels = load_pair(mat_file_path, label_file_path)
    return np.array(instances), np.array(labels)


def reformat_old_data(samples_np, labels_np, is_even):
    """sampling freq from 1000hz to 500hz"""
    count = 0
    for i in labels_np:
        count += 1
        if count % 6 == 0:
            endstr = '\n'
        else:
            endstr = ' '
        # print(i, end=endstr)

    valid_len = len(labels_np) - len(labels_np) % 6
    valid_samples = samples_np[:valid_len]
    valid_lables = labels_np[:valid_len]
    valid_samples = valid_samples.reshape(-1, 6000)
    valid_lables = valid_lables.reshape(-1, 6)
    labels = list()
    valid_rates = list()
    if is_even:
        new_samples = valid_samples[:, ::2]
    else:
        new_samples = valid_samples[:, 1::2]
    for valid_lable in valid_lables:
        # print(valid_lable)
        values, counts = np.unique(valid_lable, return_counts=True)
        label = values[np.argmax(counts)]
        labels.append(label)
        valid_rate = np.max(counts)/6
        valid_rates.append(valid_rate)
    # print(type(labels))
    # print(len(labels))
    labels_np = np.array(labels).reshape(-1, 1)
    # print(labels_np.shape)
    # print(new_samples.shape)
    data = np.hstack((new_samples, labels_np))
    return data, np.array(valid_rates)


def process_txt(file_name):
    i = 0
    que = np.zeros((5,))
    labels = list()
    with open(file_name, 'r') as f:
        next(f)
        for line in f:
            i += 1
            que[i % 6 - 1] = int(line.strip())
            if i % 6 == 0:
                endstr = '\n'
                lens = len(np.unique(que))
                labels.append(np.unique(que))
                if lens > 1:
                    raise NameError("i, len > 1")
    return np.array(labels)


if __name__ == '__main__':
    # save_train_mat()
    print(process_txt('./lables1/HypnogramAASM_subject20.txt'))
