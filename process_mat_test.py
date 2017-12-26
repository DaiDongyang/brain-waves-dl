import process_mat
import numpy as np


def test_reformat_old_data():
    mat_f = './mat1/sleep_data_row3_1.mat'
    l_f = './lables1/HypnogramAASM_subject1.txt'
    train_origin, label_origin = process_mat.load_pair_np(mat_f, l_f)
    data, valid_rate = process_mat.reformat_old_data(train_origin, label_origin, False)
    return data, valid_rate


def load_reformat_pairs():
    is_even = False
    if is_even:
        suffix = '_even'
    else:
        suffix = '_odd'
    mat_prefix = './mat1/sleep_data_row3_'
    label_prefix = './lables1/HypnogramAASM_subject'
    for i in range(1, 21):
        mat_f = mat_prefix + str(i) + '.mat'
        l_f = label_prefix + str(i) + '.txt'
        samples_np, labels_np = process_mat.load_pair_np(mat_f, l_f)
        data, valid_rates = process_mat.reformat_old_data(samples_np, labels_np, is_even)
        npy_name = './data/b_' + str(i) + suffix + '.npy'
        np.save(npy_name, data)
        # label2 = process_mat.process_txt(l_f)
        # print(data.shape)
        # print(label2.shape)
        # check = (data[:, -1] == label2[:, -1])
        # print(np.sum(check)/len(check))



if __name__ == '__main__':
    # test_reformat_old_data()
    load_reformat_pairs()
