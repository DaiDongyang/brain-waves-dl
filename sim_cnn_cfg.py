import cfg
import numpy as np

origin_d = 3000

classes = cfg.classes

# kernals
# conv_ks = [16, 32]

conv_ws = [[5, 1, 16], [5, 16, 32]]

# 0 avg_pool, 1 max_pool
pool_type = 1

pool_ksize = [1, 2, 1]

pool_strides = [1, 2, 1]

pool_padding = 'SAME'

conv_stride = 2

conv_padding = 'SAME'

# padding is same
fc_d1 = int(np.ceil(np.ceil(origin_d / pool_strides[1])/pool_strides[1])) * conv_ws[-1][-1]

fc_ds = [fc_d1, 256, 3]

keep_prob = 0.5