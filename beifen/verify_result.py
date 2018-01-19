import cfg
import numpy as np


def process_result_txt():
    results = list()
    with open(cfg.final_result_txt, 'r') as f:
        for line in f:
            result = int(line)
            results.append(result)
    g_t_m = np.load(cfg.final_test_f)
    g_t_1 = g_t_m[:, -1]
    g_t = np.array(g_t_1, dtype=int)
    p_r = np.array(results, dtype=int)
    check = (g_t == p_r)
    acc = np.sum(check)/len(check)
    print(acc)
    # g_t = g_t - 1
    # p_r = p_r - 1
    # print_csv(g_t, p_r)


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


def print_csv(gt, pr):
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
    process_result_txt()
