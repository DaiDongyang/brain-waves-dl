import os
import cfg


classes = cfg.classes

is_fft = True

is_filter_vali = False

is_filter_test = False

# norm_flag 0: no normalization; 1: standards; 2: scale
norm_flag = 2


###########################################
# config about data files
prefix_n = './origin_data/data_n'
prefix_b_even = './origin_data/data_b_even'
# prefix_b_odd = './origin_data/data_b_odd'

train_n = ['SC4001E0.npy', 'SC4041E0.npy', 'SC4081E0.npy', 'SC4121E0.npy', 'SC4162E0.npy',
           'SC4002E0.npy', 'SC4042E0.npy', 'SC4082E0.npy', 'SC4122E0.npy', 'SC4171E0.npy',
           'SC4011E0.npy', 'SC4051E0.npy', 'SC4091E0.npy', 'SC4131E0.npy', 'SC4172E0.npy',
           'SC4012E0.npy', 'SC4052E0.npy', 'SC4092E0.npy', 'SC4141E0.npy', 'SC4181E0.npy',
           'SC4021E0.npy', 'SC4061E0.npy', 'SC4101E0.npy', 'SC4142E0.npy', 'SC4182E0.npy',
           'SC4022E0.npy', 'SC4062E0.npy', 'SC4102E0.npy', 'SC4151E0.npy', 'SC4191E0.npy',
           'SC4031E0.npy', 'SC4071E0.npy', 'SC4111E0.npy', 'SC4152E0.npy', 'SC4192E0.npy',
           'SC4032E0.npy', 'SC4072E0.npy']
vali_n = ['SC4112E0.npy']
test_n = ['SC4161E0.npy']

train_b_even = ['b_1_even.npy', 'b_2_even.npy', 'b_3_even.npy', 'b_4_even.npy', 'b_5_even.npy',
                'b_6_even.npy', 'b_7_even.npy', 'b_8_even.npy', 'b_9_even.npy', 'b_10_even.npy',
                'b_11_even.npy', 'b_12_even.npy', 'b_13_even.npy', 'b_14_even.npy', 'b_15_even.npy',
                'b_16_even.npy', 'b_17_even.npy', 'b_18_even.npy']
vali_b_even = ['b_19_even.npy']
test_b_even = ['b_20_even.npy']

# train_b_odd = []
# vali_b_odd = []
# test_b_odd = []

train_n_fs = [os.path.join(prefix_n, i) for i in train_n]
vali_n_fs = [os.path.join(prefix_n, i) for i in vali_n]
test_n_fs = [os.path.join(prefix_n, i) for i in test_n]

train_b_e_fs = [os.path.join(prefix_b_even, i) for i in train_b_even]
vali_b_e_fs = [os.path.join(prefix_b_even, i) for i in vali_b_even]
test_b_e_fs = [os.path.join(prefix_b_even, i) for i in test_b_even]

train_fs = train_n_fs + train_b_e_fs
vali_fs = vali_n_fs + vali_b_e_fs
test_fs = test_n_fs + test_b_e_fs

# above config about data file ###
##########################################
