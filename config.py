import utils

class xworld_config:
    ##### Environment parameters
    env_name = 'xworld'

    vocab_dir = './data'
    conf_path = 'data/navigation.json'
    eval_conf_path = 'data/navigation.json'

    curriculum = 10000
    need_mem_cmd = 0

    grid_size = 13
    pixel_per_grid = 12
    img_size = pixel_per_grid * grid_size

    num_actions = 4

    ##### Language Module
    word_embed_dim  = 1024
    synt_embed_dim  = 128
    func_embed_dim  = 128
    lang_hidden_dim = 128

    program_steps   = 3

    ##### Action Module
    act_hidden_dim  = 512
    act_filter_size = 3
    act_num_filters = [num_actions * 16, num_actions]
    act_padding     = act_filter_size // 2

    ##### Perception Module
    feat_visual_dim  = word_embed_dim // 2
    feat_spatial_dim = word_embed_dim - feat_visual_dim

    vis_num_filters  = [64, 64, feat_visual_dim, feat_visual_dim]
    vis_filter_sizes = [3, 2, 2]
    vis_strides      = [3, 2, 2]
    ffs, fs = utils.compute_block_alignment(vis_filter_sizes, vis_strides, pixel_per_grid)
    vis_filter_sizes.append(ffs)
    vis_strides.append(fs)

    ##### Reinforcement learning
    explore_frame = 500000         # number of frames using epsilon greedy exploration
    alpha = 1.0                    # epsilon greedy

    replay_size = 10000            # max size of replay memory
    init_size = 1000               # number of memory entries from which the training starts
    prioritize = False
    beta0 = 0.5

    gamma = 0.99                   # discount factor

    ##### Optimization
    batch_size = 16
    train_interval = 4

    algo = 'rmsprop'
    lr   = 1e-5
    mom  = 0.9
    w_decay = 0.

    ##### Monitoring
    monitor_gnorm = False
    monitor_mask  = False
