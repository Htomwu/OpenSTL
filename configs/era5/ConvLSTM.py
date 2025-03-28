from configs.era5.TAU import epochs

method = 'ConvLSTM'
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0
# model
num_hidden = '16,16'
filter_size = 3
stride = 1
patch_size = 1
layer_norm = False
# training
lr = 0.001