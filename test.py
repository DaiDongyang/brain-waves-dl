import data_set
import numpy as np

def test_DataSet():
    samples = np.array([[1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3],
                        [1, 4],
                        [1, 5],
                        [1, 6]])
    labels = np.array([1, 2, 3, 4, 5, 6, 7])

    d_s = data_set.DataSet(samples, None)
    print_samples_ls(d_s.get_samples(), d_s.get_labels())

    batch_size = 2
    print()
    print()
    for i in range(5):
        samples_batch, ls_batch, is_epoch_end = d_s.next_batch_fix2(batch_size)
        print_samples_ls(samples_batch, ls_batch)
        print(is_epoch_end)
        print()
        samples_batch, ls_batch, is_epoch_end = d_s.next_batch_fix2_order(batch_size)
        print_samples_ls(samples_batch, ls_batch)
        print(is_epoch_end)
        print()


def print_samples_ls(samples, ls):
    # print(ls.shape)
    for i in zip(samples, ls):
        print(i)
    # print()


if __name__ == '__main__':
    test_DataSet()