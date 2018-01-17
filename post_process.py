import pickle
import numpy as np
import cfg
from collections import Counter


def dump_list(l, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(l, pickle_f)


def load_pickle(pickle_name):
    r = None
    with open(pickle_name, 'rb') as pickle_f:
        r = pickle.load(pickle_f)
    return r


def process_results(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    # print(Counter(gt))
    # print(Counter(pr))
    # print(Counter(zip(gt, pr)))
    for i, j in zip(gt, pr):
        matrix[i, j] += 1
    # print(matrix)
    return matrix, classes


def print_csv():
    gt = load_pickle(cfg.gt_pickle)
    pr = load_pickle(cfg.pr_pickle)
    total_acc = float(np.sum(np.array(gt) == np.array(pr)) / len(gt))
    matrix, classes = process_results(gt, pr, cfg.classes)
    print()
    print('  a\\p', end='\t')
    for c in classes:
        print(c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ',classes[i], end='\t')
        for ele in matrix[i]:
            print(ele, end='\t')
        print()
    print()

    sum_1 = np.sum(matrix, axis=1)
    matrix2 = matrix / sum_1.reshape((-1, 1))

    print('  a\\p', end='\t')
    for c in classes:
        print(' ', c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ', classes[i], end='\t')
        for ele in matrix2[i]:
            print('%.4f' % ele, end='\t')
        print()
    print()

    avg = 0
    for i in range(len(classes)):
        avg += matrix2[i, i]
    print('  average accurate is %.4f' % (avg/len(classes)))
    print('  total accurate is %.4f' % total_acc)
    print()


if __name__ == '__main__':
    print_csv()
