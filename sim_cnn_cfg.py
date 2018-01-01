import cfg
import numpy as np

visible_device = '2'

batch_size = 500

loop_epoch_num = 100000

log_epoch_num = 50

# avoid rewrite
persist_epoch_num = 50

save_epoch_num = 5

learning_rate = 1e-4

# Consistent with is_fft
origin_d = 1500

classes = cfg.classes

# kernals
# conv_ks = [16, 32]

conv_ws = [[10, 1, 16], [10, 16, 32]]

# 0 avg_pool, 1 max_pool
pool_type = 1

pool_ksize = [2]

pool_strides = [2]

pool_padding = 'SAME'

conv_stride = 1

conv_padding = 'SAME'

# padding is same
fc_d1 = int(np.ceil(np.ceil(origin_d / pool_strides[0])/pool_strides[0])) * conv_ws[-1][-1]

fc_ds = [fc_d1, 256, 3]

keep_prob = 0.5
