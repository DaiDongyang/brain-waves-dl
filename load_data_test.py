import load_data
import numpy as np
import cfg
from collections import Counter


def test_filter_single_npy():
    f = './origin_data/data_b_even/b_1_even.npy'
    data_origin = np.load(f)
    data_filter = load_data.filter_single_subset(data_origin, cfg.classes)
    print(data_origin.shape)
    print(np.unique(data_origin[:, -1]))
    print(data_filter.shape)
    print(np.unique(data_filter[:, -1]))


def test_filter_files():
    fs = ['./origin_data/data_b_even/b_1_even.npy',
          './origin_data/data_b_even/b_2_even.npy',
          './origin_data/data_b_even/b_3_even.npy']
    dataset = load_data.load_files(fs, cfg.classes)
    print(dataset.shape)
    print(np.unique(dataset[:, -1]))
    data, ls = load_data.div_samples_labels(dataset)
    ls_1hot = load_data.convert_ls_1hot(ls, cfg.classes)
    print(Counter(ls))
    print(cfg.classes)
    print(np.sum(ls_1hot, axis=0))
    # print(data.shape)
    # print(ls.shape)
    # print(np.unique(ls))


def test_TrainValiTest():
    tvt = load_data.TrainValiTest()
    tvt.load()
    train_samples, train_ls  = tvt.train_samples_ls()
    vali_samples, vali_ls = tvt.vali_samples_ls()
    test_samples, test_ls = tvt.test_samples_ls()
    print(train_samples.shape, train_ls.shape)
    print('mu', np.average(train_samples, axis=0))
    print('sigma', np.std(train_samples, axis=0))
    print('max', np.max(train_samples, axis=0))
    print('min', np.min(train_samples, axis=0))
    print(vali_samples.shape, vali_ls.shape)
    print('mu', np.average(vali_samples, axis=0))
    print('sigma', np.std(vali_samples, axis=0))
    print('max', np.max(vali_samples, axis=0))
    print('min', np.min(vali_samples, axis=0))
    print(test_samples.shape, test_ls.shape)
    print('mu', np.average(test_samples, axis=0))
    print('sigma', np.std(test_samples, axis=0))
    print('max', np.max(test_samples, axis=0))
    print('min', np.min(test_samples, axis=0))


if __name__ == '__main__':
    test_TrainValiTest()
    # test_filter_files()
    # test_filter_single_npy()