import sys, os
import argparse
from collections import OrderedDict, defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp

import torchvision.utils as vutils
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from py_simulator import Simulator
from config import xworld_config, navonly_config
from model import Agent
from vocab import Vocab
from replay_memory import *
from functions import *
from utils import *

config = xworld_config()
fig, axes = plt.subplots(6,5, figsize=(36, 20))
fig.tight_layout(pad=5, w_pad=2, h_pad=5)

def visualize_spatial_attn(save_dir, info):
    image = opencv_to_rgb(info['image'])
    image[:,:,6*12+3:7*12-3,6*12+3:7*12-3] = image.min()
    vutils.save_image(image, os.path.join(save_dir, 'image.png'), nrow=6, pad_value=1, padding=1)

    env_map_vis = vis_scale_image(info['env_map'].data.cpu(), config.pixel_per_grid)
    vutils.save_image(env_map_vis, os.path.join(save_dir, 'env_map.png'), nrow=6, pad_value=1, padding=1)

    grid_attn_vis = vis_scale_image(info['grid_attns'].data.cpu(), config.pixel_per_grid)
    vutils.save_image(grid_attn_vis, os.path.join(save_dir, 'grid_attn.png'), normalize=False, scale_each=False, nrow=6, pad_value=1, padding=1)

    heatmap_vis = vis_scale_image(info['heatmaps'].data.cpu(), config.pixel_per_grid)
    vutils.save_image(heatmap_vis, os.path.join(save_dir, 'heatmap.png'), normalize=False, scale_each=False, nrow=6, pad_value=1, padding=1)

    cached_attn_vis = vis_scale_image(info['cached_attns'].data.cpu(), config.pixel_per_grid)
    vutils.save_image(cached_attn_vis, os.path.join(save_dir, 'cached_attn.png'), normalize=False, scale_each=False, nrow=6, pad_value=1, padding=1)

def visualize_sequence_attn(save_dir, vocab, info, axes, fig, task):
    clear_axes(axes)
    seq_attns = info['seq_attns'].permute(1, 0, 2).cpu().numpy()  # [batch x step x seqlen]
    sequence = info['question'].numpy() if task == 'rec' else info['command'].numpy()
    for idx in range(seq_attns.shape[0]):
        i, j = idx / 5, idx % 5
        seq_attn = seq_attns[idx]
        xticks = vocab.convert_to_sym(sequence[:,idx])
        yticks = ['step {}'.format(step) for step in range(seq_attn.shape[0])]
        if task == 'rec':
            answer, predict = vocab.convert_to_sym([info['answer'][idx], info['predict'][idx]])
            axes[i,j].set_title('A: {} <-> P: {}'.format(answer, predict))
        vis_seq_attn(axes[i,j], seq_attn, xticks, yticks)
    fig.savefig(os.path.join(save_dir, 'seq_attn.png'), format='png')

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def visualize_all_steps(nav_info_list, vocab):
    save_dir = os.path.join('evaluate', str(time.time()))
    mkdir(save_dir)
    
    info = {}
    info['image'] = torch.cat([nav_info['image'] for nav_info in nav_info_list])
    info['env_map'] = torch.cat([nav_info['env_map'] for nav_info in nav_info_list])
    info['grid_attns'] = torch.cat([nav_info['grid_attns'][-1] for nav_info in nav_info_list])
    info['heatmaps'] = torch.cat([nav_info['heatmaps'][-1] for nav_info in nav_info_list])
    info['cached_attns'] = torch.cat([nav_info['cached_attns'][-1] for nav_info in nav_info_list])
    visualize_spatial_attn(save_dir, info)

    info['seq_attns'] = torch.cat([torch.stack(nav_info['seq_attns']).data.cpu() for nav_info in nav_info_list], dim=1)
    info['command'] = torch.cat([nav_info['command'] for nav_info in nav_info_list], dim=1)
    visualize_sequence_attn(save_dir, vocab, info, axes, fig, task='nav')

def run_episode(env, act_net, vocab):
    env.reset_game()
    command = None
    success_flag = 0.

    rewards = []
    nav_info_list = []
    while True:
        curr_state = xwd_get_state(config, env, vocab, command)

        done = env.game_over()
        if done != 'alive':
            if config.show_screen: env.show_screen()
            episode_return = sum(rewards)

            if 'success' not in done and len(nav_info_list) > 0:
                if config.vis_fail: visualize_all_steps(nav_info_list, vocab)
            else:
                success_flag = 1.

            break

        image, command, question, answer = curr_state

        if command is None:
            action = xwd_random_step(env)
        else:
            if config.show_screen: env.show_screen()
            action, nav_info = act_net(variable(image), command=variable(command), act_only=True)
            action = action.data[0,0]
            reward = env.take_actions({'action': action, 'pred_sentence': ''})
            rewards.append(float(reward))

            nav_info['image'] = image
            nav_info['command'] = command 
            nav_info_list.append(nav_info)

    return episode_return, success_flag

def main():
    checkpoint = torch.load(latest_checkpoint(os.path.join(config.load_dir, 'checkpoint')))
    vocab = checkpoint['vocab']

    model = Agent(config, vocab.size())
    model.load_state_dict(checkpoint['model'])

    env = Simulator.create(config.env_name, {'conf_path':config.eval_conf_path, 'curriculum':0, 
                                                  'task_mode':'arxiv_lang_acquisition'})
    if config.cuda:
        model.cuda()
    model.eval()

    total_return, total_success = 0., 0.

    for i in range(config.eval_episode):
        ep_return, ep_success= run_episode(env, model, vocab)
        total_return += ep_return
        total_success += ep_success

    print('Average episode return: {:.4f}; Success rate {:.3f}'.format(float(total_return) / config.eval_episode, float(total_success) / config.eval_episode))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch xworld evaluation')
    parser.add_argument('--eval_episode', type=int, default=100, metavar='N',
                        help='number of episodes to run')
    parser.add_argument('--load_dir', type=str, default='log_xworld', 
                        help='directory to load checkpoints')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--show_screen', action='store_true', default=False,
                        help='shows screen for sanity check')
    parser.add_argument('--vis_fail', action='store_true', default=False,
                        help='visualize failure cases')
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        setattr(config, k, v)

    global variable
    variable = create_variable_func(config.cuda)
    
    main()
