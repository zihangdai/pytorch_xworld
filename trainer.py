from __future__ import print_function, division
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
from config import xworld_config
from model import Agent
from vocab import Vocab
from replay_memory import *
from functions import *
from utils import *

config = xworld_config()

rec_fig, rec_axes = plt.subplots(4,4, figsize=(24, 15))
rec_fig.tight_layout(pad=5, w_pad=2, h_pad=5)

nav_fig, nav_axes = plt.subplots(4,4, figsize=(24, 15))
nav_fig.tight_layout(pad=5, w_pad=2, h_pad=5)

def train_recognition(act_net, rec_memory):
    # ========== Supervised Learning ==========
    rec_batch = rec_memory.sample(config.batch_size)
    rec_image, question, answer = rec_batch

    rec_logit, rec_info = act_net(variable(rec_image), question=variable(question))
    ce_loss = F.cross_entropy(rec_logit, variable(answer))

    predict = rec_logit.max(1, **kwargs)[1].squeeze(1).data.cpu()
    rec_accu = torch.eq(predict, answer).float().mean()

    # ========== Monitoring ==========
    rec_info['rec_accu'] = rec_accu
    rec_info['image'] = rec_image
    rec_info['question'] = question
    rec_info['predict'] = predict
    rec_info['answer'] = answer

    return ce_loss, rec_info

def train_prioritize(act_net, tgt_net, nav_memory):
    # ========== Reinforcement Learning ===========
    nav_batch, nav_weight, nav_indices = nav_memory.sample(config.batch_size, config.beta)
    curr_state, action, next_state, reward = nav_batch
    nav_weight = variable(nav_weight.unsqueeze(1))
    
    curr_image, curr_command = curr_state
    _, action_prob, curr_value, nav_info = act_net(variable(curr_image), command=variable(curr_command))

    next_image, next_command = next_state
    next_value, _ = tgt_net(variable(next_image, volatile=True), command=variable(next_command, volatile=True), val_only=True)
    next_value.volatile = False

    # This essentially implements the Huber loss
    target_value = config.gamma * next_value + variable(reward)
    td_error = torch.clamp(target_value - curr_value, -1., 1.).detach()
    td_loss = torch.mean(-td_error * curr_value * nav_weight)

    log_prob = torch.log(1e-6 + action_prob)
    act_log_prob = log_prob.gather(1, variable(action.unsqueeze(1)))
    pg_loss = torch.mean(-td_error * act_log_prob * nav_weight)

    nav_priorities = (torch.abs(td_error.squeeze(1)).data + 1e-6).cpu().tolist()
    nav_memory.update_priorities(nav_indices, nav_priorities)

    # ========== Monitoring ==========
    nav_info['command'] = curr_command
    nav_info['image'] = curr_image
    nav_info['weight'] = nav_weight
    nav_info['td_error'] = td_error.data.abs().mean()

    return td_loss, pg_loss, nav_info

def train_standard(act_net, tgt_net, nav_memory):
    # ========== Reinforcement Learning ===========
    curr_state, action, next_state, reward = nav_memory.sample(config.batch_size)
    
    curr_image, curr_command = curr_state
    _, action_prob, curr_value, nav_info = act_net(variable(curr_image), command=variable(curr_command))

    next_image, next_command = next_state
    next_value, _ = tgt_net(variable(next_image, volatile=True), command=variable(next_command, volatile=True), val_only=True)
    next_value.volatile = False

    # This essentially implements the Huber loss
    target_value = variable(reward) + config.gamma * next_value
    td_error = torch.clamp(target_value - curr_value, -1., 1.).detach()
    td_loss = torch.mean(-td_error * curr_value)

    log_prob = torch.log(1e-6 + action_prob)
    act_log_prob = log_prob.gather(1, variable(action.unsqueeze(1)))
    pg_loss = torch.mean(-td_error * act_log_prob)

    # ========== Monitoring ==========
    nav_info['command'] = curr_command
    nav_info['image'] = curr_image
    nav_info['td_error'] = td_error.data.abs().mean()

    return td_loss, pg_loss, nav_info

def visualize_spatial_attn(save_dir, info, task):
    vutils.save_image(opencv_to_rgb(info['image'].cpu()), os.path.join(save_dir, 'image.png'), nrow=4, pad_value=1, padding=1)
    if task == 'nav':
        env_map_vis = vis_scale_image(info['env_map'].data.cpu(), config.pixel_per_grid)
        vutils.save_image(env_map_vis, os.path.join(save_dir, 'env_map.png'), normalize=False, scale_each=False, nrow=4, pad_value=1, padding=1)

    grid_attn_vis = vis_scale_image(info['grid_attns'][-1].data.cpu(), config.pixel_per_grid)
    vutils.save_image(grid_attn_vis, os.path.join(save_dir, 'grid_attn.png'), normalize=False, scale_each=False, nrow=4, pad_value=1, padding=1)

    heatmap_vis = vis_scale_image(info['heatmaps'][-1].data.cpu(), config.pixel_per_grid)
    vutils.save_image(heatmap_vis, os.path.join(save_dir, 'heatmap.png'), normalize=False, scale_each=False, nrow=4, pad_value=1, padding=1)

    cached_attn_vis = vis_scale_image(info['cached_attns'][-1].data.cpu(), config.pixel_per_grid)
    vutils.save_image(cached_attn_vis, os.path.join(save_dir, 'cached_attn.png'), normalize=False, scale_each=False, nrow=4, pad_value=1, padding=1)

def visualize_sequence_attn(save_dir, vocab, info, axes, fig, task):
    clear_axes(axes)
    seq_attns = torch.stack(info['seq_attns']).data.permute(1, 0, 2).cpu().numpy()  # [batch x step x seqlen]
    sequence = info['question'].numpy() if task == 'rec' else info['command'].numpy()
    for idx in range(seq_attns.shape[0]):
        i, j = idx // 4, idx % 4
        seq_attn = seq_attns[idx]
        xticks = vocab.convert_to_sym(sequence[:,idx])
        yticks = ['step {}'.format(step) for step in range(seq_attn.shape[0])]
        if task == 'rec':
            answer, predict = vocab.convert_to_sym([info['answer'][idx], info['predict'][idx]])
            axes[i,j].set_title('A: {} <-> P: {}'.format(answer, predict))
        vis_seq_attn(axes[i,j], seq_attn, xticks, yticks)
    fig.savefig(os.path.join(save_dir, 'seq_attn.png'), format='png')

def train_model(act_net, tgt_net, rec_memory, nav_memory, optimizer, vocab, vis=False):
    act_net.train()
    # ========== Forward computation ==========
    if config.prioritize:
        td_loss, pg_loss, nav_info = train_prioritize(act_net, tgt_net, nav_memory)
    else:
        td_loss, pg_loss, nav_info = train_standard(act_net, tgt_net, nav_memory)

    ce_loss, rec_info = train_recognition(act_net, rec_memory)

    # ========== Optimization & Monitoring ==========
    if config.monitor_gnorm:
        optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        gnorm_td = grad_norm(act_net.parameters())

        optimizer.zero_grad()
        pg_loss.backward(retain_variables=True)
        gnorm_pg = grad_norm(act_net.parameters())

        optimizer.zero_grad()
        ce_loss.backward(retain_variables=True)
        gnorm_ce = grad_norm(act_net.parameters())
    
    tot_loss = td_loss + pg_loss + ce_loss
    
    optimizer.zero_grad()
    tot_loss.backward()
    optimizer.step()

    if config.monitor_gnorm:
        gnorm_tot = grad_norm(act_net.parameters())

    if vis:
        # ========== recognition task ==========
        visualize_spatial_attn(os.path.join(config.save_dir, 'vis_rec'), rec_info, task='rec')

        visualize_sequence_attn(os.path.join(config.save_dir, 'vis_rec'), vocab, rec_info, rec_axes, rec_fig, task='rec')

        # ========== navigation task ==========
        visualize_spatial_attn(os.path.join(config.save_dir, 'vis_nav'), nav_info, task='nav')

        visualize_sequence_attn(os.path.join(config.save_dir, 'vis_nav'), vocab, nav_info, nav_axes, nav_fig, task='nav')

    return_dict = {}
    return_dict.update({'loss ce' : ce_loss.data[0], 'rec accu' : rec_info['rec_accu']})
    return_dict.update({'loss td' : td_loss.data[0], 'loss pg' : pg_loss.data[0], 'abs td error':nav_info['td_error']})

    if config.monitor_mask:
        return_dict['rec mask mean'] = rec_info['rec_mask'].data.mean()
        return_dict['rec mask std'] = rec_info['rec_mask'].data.std(1).mean()
        for step in range(config.program_steps):
            return_dict['mask {} std'.format(step)]   = rec_info['masks'][step].data.std(1).mean()
            return_dict['mask {} mean'.format(step)]  = rec_info['masks'][step].data.mean()
            return_dict['sigma {} mean'.format(step)] = rec_info['sigmas'][step].data.mean()

    if config.monitor_gnorm:
        return_dict.update({'gnorm ce' : gnorm_ce})
        return_dict.update({'gnorm td' : gnorm_td, 'gnorm pg' : gnorm_pg})
        return_dict.update({'gnorm tot': gnorm_tot})

    return return_dict

def evaluate(env, act_net, vocab):
    act_net.eval()

    env.reset_game()
    command = None

    rewards = []
    while True:
        curr_state = xwd_get_state(config, env, vocab, command)
        if config.show_screen: env.show_screen()

        done = env.game_over()
        if done != 'alive':
            if config.show_screen: env.show_screen()
            print('=====> Eval episode done with status {} and total return {}'.format(done, sum(rewards)))
            break

        image, command, question, answer = curr_state

        if command is None:
            action = xwd_random_step(env)
        else:
            action, _ = act_net(variable(image, volatile=True), command=variable(command, volatile=True), act_only=True)
            reward = env.take_actions({'action': action.data[0,0], 'pred_sentence': ''})
            rewards.append(float(reward))

def main():
    
    env = Simulator.create(config.env_name, {'conf_path':config.conf_path, 'curriculum':config.curriculum,
                                             'task_mode':'arxiv_lang_acquisition'})

    eval_env = Simulator.create(config.env_name, {'conf_path':config.eval_conf_path, 'curriculum':0, 
                                                  'task_mode':'arxiv_lang_acquisition'})
    
    log_experiment_config(config)
    create_log_dir(root_dir=config.save_dir, sub_dirs=['checkpoint', 'vis_rec', 'vis_nav'])

    vocab = load_vocab(config.vocab_dir, special_syms=['$', '#oov#'])
    if config.prioritize:
        nav_memory = PrioritizedReplayMemory(capacity=config.replay_size, pack_func=pack_batch_nav)
    else:
        nav_memory = ReplayMemory(capacity=config.replay_size, pack_func=pack_batch_nav)
    rec_memory = ReplayMemory(capacity=config.replay_size//4, pack_func=pack_batch_rec)
    
    act_net = Agent(config, vocab.size())
    tgt_net = Agent(config, vocab.size())

    if config.cuda:
        act_net.cuda()
        tgt_net.cuda()
    copy_state(act_net, tgt_net)
    print('=====> Update target network; Current explore alpha {:.3f}'.format(act_net.alpha))
    
    if config.algo == 'rmsprop':
        optimizer = torch.optim.RMSprop(act_net.parameters(), lr=config.lr, momentum=config.mom, weight_decay=config.w_decay)
    elif config.algo == 'adagrad':
        optimizer = torch.optim.Adagrad(act_net.parameters(), lr=config.lr, weight_decay=config.w_decay)
    elif config.algo == 'adam':
        optimizer = torch.optim.Adam(act_net.parameters(), lr=config.lr, betas=(config.mom, 0.9999), weight_decay=config.w_decay)
    else:
        raise ValueError('Unsupported optimization algorithm {}. Please choose from ["rmsprop", "adagrad", "adam"].'.format(config.algo))

    monitor, done_states = Monitor(track_time=True), Monitor(count=False, default_val=0)
    
    train_cnt, frame_cnt = 0, 0
    for eidx in range(config.max_episode):
        # Annealing hyper-parameters
        act_net.alpha = max(0., config.alpha * (1. - float(frame_cnt) / float(config.explore_frame)))
        tgt_net.alpha = max(0., config.alpha * (1. - float(frame_cnt) / float(config.explore_frame)))
        config.beta = min(1., config.beta0 + float(eidx) / float(config.max_episode / 2) * (1. - config.beta0))

        # Get initial state
        env.reset_game()
        command = None

        # Ignore the first few empty frames without any instruction
        curr_state = xwd_get_state(config, env, vocab, command)

        # Episode loop
        rewards = []
        while True:
            # If not alive, exit the loop
            done = env.game_over()
            if done != 'alive':
                values = reward_to_value(rewards, config.gamma)
                
                if len(values) > 0:
                    monitor.update('emp_value', values[0])
                    monitor.update('steps', len(rewards))
                done_states.update(done, 1)

                break

            # Unpack the current state into specific
            image, command, question, answer = curr_state

            # Push image, question, answer into the recognition memory
            # - NOTE: Recognition task is treated as an independent channel at this moment.
            #         As a result, as long as the current state and question and answer, we
            #         record an entry in the memory
            if question is not None and answer is not None:
                entry = image, question, answer
                rec_memory.push(entry)

            # When the command is None, it means the navigation task has not started yet.
            # In this case, just take a random step
            if command is None:
                action = xwd_random_step(env)
            else:
                action, _ = act_net(variable(image, volatile=True), command=variable(command, volatile=True), act_only=True)
                action = action.data[0,0]
                reward = env.take_actions({'action': action, 'pred_sentence': ''})
                rewards.append(float(reward))
                frame_cnt += 1

            # Get the next state
            next_state = xwd_get_state(config, env, vocab, command)

            # Store an effective navigation transition in navigation memory
            if command is not None:
                entry = (curr_state[0], curr_state[1]), action, (next_state[0], next_state[1]), reward
                nav_memory.push(entry)

            # Move to the next state
            curr_state = next_state
            
            # Train the act_net with a mini-batch sampled from the replay memory
            if frame_cnt > 0 and frame_cnt % config.train_interval == 0 and \
               len(nav_memory) >= config.init_size and \
               len(rec_memory) >= config.init_size//4:

                iter_vals = train_model(act_net, tgt_net, rec_memory, nav_memory, optimizer, vocab, vis=(train_cnt % config.log_interval == 0))
                monitor.update_dict(iter_vals)
                train_cnt += 1
                
            # Log training info
            if train_cnt > 0 and train_cnt % config.log_interval == 0 and frame_cnt % config.train_interval == 0:
                disp_str = '#{} {}'.format(train_cnt, eidx)
                disp_str += monitor.disp(reset=True)
                disp_str += done_states.disp(reset=True)

                print(disp_str)

            # Update target network parameters
            if train_cnt > 0 and train_cnt % 2000 == 0 and frame_cnt % config.train_interval == 0:
                copy_state(act_net, tgt_net)
                print('=====> Update target network; Current explore alpha {:.3f}'.format(act_net.alpha))

        # Sanity check
        if eidx > 0 and eidx % config.eval_interval == 0:
            save_checkpoint(os.path.join(config.save_dir, 'checkpoint', 'chk.ep{}'.format(eidx)), act_net, optimizer, vocab)
            evaluate(eval_env, act_net, vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch xworld training')
    parser.add_argument('--max_episode', type=int, default=500000, metavar='N',
                        help='maximum episode (default: 500000)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='interval between training status logs (default: 200)')
    parser.add_argument('--eval_interval', type=int, default=500, metavar='N',
                        help='interval between evaluations (default: 500)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='mini batch size used for training(default: 16)')
    parser.add_argument('--save_dir', type=str, default='log_xworld', 
                        help='directory to save intermediate results')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--prioritize', action='store_true', default=False,
                        help='enables prioritized memory replay')
    parser.add_argument('--show_screen', action='store_true', default=False,
                        help='shows screen for sanity check')
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        setattr(config, k, v)
    
    global variable
    variable = create_variable_func(config.cuda)

    main()
