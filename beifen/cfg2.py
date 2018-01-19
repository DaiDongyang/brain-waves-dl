import os

id_str = 'reduce_01191453'

is_log_cfg = False

# ================================================
# GPU
visible_device = '3'
# per_process_gpu_memory_fraction = 1

# =================================================

batch_size = 256

loop_epoch_nums = [100, 100, 100, 100]

learning_rates = [0.1, 0.05, 0.02, 0.01]

log_epoch_num = 5

# learning_rate = 0.05

dropout_probs = [1, 0.2, 0.5]

optimizer_type = 'grad'     # 'adam', 'adadelta', 'grad'


origin_d = 1500

# 0, cross entropy;  1, weight loss; 2, focal_loss
loss_type = 0

loss_weights = [1, 1, 1]


# ==================================================
# checkpoint and restore
only_run_final_test = False

final_result_txt = 'result.txt'


tvt_params_pickle_name = './pickles/tvt_params.pickle'

is_train = True

is_restore = False

restore_file = 'p-my-model1/p-my-model01191304_400'

restart_epoch_i = 0

persist_checkpoint_interval = 200

persist_checkpoint_file = 'p-my-model/p-my-model' + id_str + '_'

# ==================================================
# result pickle

gt_pickle = './pickles/gt_' + id_str + '.pickle'

pr_pickle = './pickles/pr_' + id_str + '.pickle'


# ==================================================
# config about data

classes = [1, 2, 3]

is_fft = True

fft_clip = -1

is_filter_vali = False

is_filter_test = False

features_reduce = 0  # 0, no reduce, 1, pca

reduce_dim = 200

# norm_flag 0: no normalization; 1: standards; 2: scale
norm_flag = 1

###########################################
# config about data files
final_test_f = './origin_data/data_n/SC4081E0.npy'

prefix_n = './origin_data/data_n'
prefix_b_even = './origin_data/data_b_even'
prefix_b_odd = './origin_data/data_b_odd'

train_n = ['SC4121E0.npy', 'SC4162E0.npy', 'SC4032E0.npy', 'SC4081E0.npy', 'SC4072E0.npy',
           'SC4002E0.npy', 'SC4042E0.npy', 'SC4082E0.npy', 'SC4122E0.npy', 'SC4171E0.npy',
           'SC4011E0.npy', 'SC4051E0.npy', 'SC4091E0.npy', 'SC4131E0.npy', 'SC4172E0.npy',
           'SC4052E0.npy', 'SC4092E0.npy', 'SC4141E0.npy', 'SC4181E0.npy',
           'SC4021E0.npy', 'SC4061E0.npy', 'SC4101E0.npy', 'SC4142E0.npy', 'SC4182E0.npy',
           'SC4022E0.npy', 'SC4151E0.npy', 'SC4191E0.npy', 'SC4031E0.npy', 'SC4112E0.npy',
           'SC4111E0.npy', 'SC4152E0.npy', 'SC4192E0.npy',  'SC4102E0.npy', ]
vali_n = ['SC4062E0.npy', 'SC4012E0.npy', ]
test_n = ['SC4161E0.npy', 'SC4071E0.npy', 'SC4001E0.npy', 'SC4041E0.npy', ]

train_b_even = ['b_1_even.npy', 'b_2_even.npy', 'b_3_even.npy', 'b_4_even.npy',
                'b_5_even.npy', 'b_6_even.npy', 'b_7_even.npy', 'b_8_even.npy',
                'b_9_even.npy', 'b_10_even.npy', 'b_11_even.npy', 'b_12_even.npy',
                'b_13_even.npy', 'b_14_even.npy', 'b_15_even.npy','b_16_even.npy',
                'b_17_even.npy', 'b_18_even.npy', 'b_19_even.npy', 'b_20_even.npy']

# train_b_even = []
vali_b_even = []
test_b_even = []

# train_b_odd = ['b_1_odd.npy', 'b_2_odd.npy', 'b_3_odd.npy', 'b_4_odd.npy', 'b_5_odd.npy',
#                'b_6_odd.npy', 'b_7_odd.npy', 'b_8_odd.npy', 'b_9_odd.npy', 'b_10_odd.npy',
#                'b_11_odd.npy', 'b_12_odd.npy', 'b_13_odd.npy', 'b_14_odd.npy', 'b_15_odd.npy',
#                'b_16_odd.npy', 'b_17_odd.npy', 'b_18_odd.npy', 'b_19_odd.npy', 'b_20_odd.npy']
train_b_odd = []
vali_b_odd = []
test_b_odd = []

train_n_fs = [os.path.join(prefix_n, i) for i in train_n]
vali_n_fs = [os.path.join(prefix_n, i) for i in vali_n]
test_n_fs = [os.path.join(prefix_n, i) for i in test_n]

train_b_e_fs = [os.path.join(prefix_b_even, i) for i in train_b_even]
vali_b_e_fs = [os.path.join(prefix_b_even, i) for i in vali_b_even]
test_b_e_fs = [os.path.join(prefix_b_even, i) for i in test_b_even]

train_b_o_fs = [os.path.join(prefix_b_odd, i) for i in train_b_odd]
vali_b_o_fs = [os.path.join(prefix_b_odd, i) for i in vali_b_odd]
test_b_o_fs = [os.path.join(prefix_b_odd, i) for i in test_b_odd]


train_fs = train_n_fs + train_b_e_fs
vali_fs = vali_n_fs + vali_b_e_fs
test_fs = test_n_fs + test_b_e_fs

# above config about data file ###
##########################################

# ================================================
# config for cnn

conv_fs = [[10, 1, 16], [10, 16, 32]]

conv_stride = 1

conv_padding = 'SAME'

# if cnn_pool_type == 0, avg_pool; else max_pool
cnn_pool_type = 1

cnn_pool_padding = 'SAME'

cnn_pool_ksize = [4]

cnn_pool_strides = [2]


# ===============================================
# config for rnn

# rnn_seq_d = 25

# rnn_units_list = [128]
rnn_units_list = []

rnn_is_bidirection = True

rnn_cell_type = 'gru'  # 'gru', 'block_lstm'

# ===============================================

fc_w_shapes = [[int(1500*32/4), 64], [64, 3]]

# ===============================================
# config for nn
