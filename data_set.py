import numpy as np


class DataSet:

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        size = len(samples)
        if size != len(labels):
            raise NameError('size not equal for samples and labels')
        self.data_size = size
        idxes = np.array(range(size))
        np.random.shuffle(idxes)
        # random order for all indexes
        self.idxes = idxes
        self.next_iidx = 0

    def re_shuffle(self):
        np.random.shuffle(self.idxes)

    # if the size of the last batch for an epoch is less than batch_size, return a less batch
    def next_batch_fix2(self, batch_size):
        is_epoch_end = False
        end = self.next_iidx + batch_size
        if end >= self.data_size:
            is_epoch_end = True
            end = self.data_size
        iidxes = np.arange(self.next_iidx, end)
        self.next_iidx = end % self.data_size
        batch_idxes = self.idxes[iidxes]
        return self.samples[batch_idxes], self.labels[batch_idxes], is_epoch_end

    def next_batch_fix(self, batch_size):
        samples, labels, is_epoch_end = self.next_batch_fix2(batch_size)
        if is_epoch_end:
            self.re_shuffle()
        return samples, labels

    def next_batch(self, batch_size):
        end = self.next_iidx + batch_size
        iidx = np.arange(self.next_iidx, end)
        if end > self.data_size:
            iidx = iidx % self.data_size
            end = end % self.data_size
        self.next_iidx = end
        batch_idxes = self.idxes[iidx]
        return self.samples[batch_idxes], self.labels[batch_idxes]

    def batch_num(self, batch_size):
        return int(np.ceil(self.data_size / batch_size))

    def get_samples(self):
        return self.samples[self.idxes]

    def get_labels(self):
        return self.labels[self.idxes]

#
# class EvalQue:
#     pass
