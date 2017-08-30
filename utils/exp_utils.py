import os, sys
import shutil
from glob import glob
from collections import OrderedDict, Sequence

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable

import numpy as np

from datetime import datetime
import time
from vocab import Vocab

if sys.version_info >= (3,0):
    raw_input = input

class Monitor(object):
    def __init__(self, count=True, default_val=0., track_time=False):
        self.count = count
        self.default_val = default_val
        self.track_time = track_time
        
        self.reset()

    def update(self, key, val):
        if key not in self.monitor:
            self.monitor[key] = self.default_val
        self.monitor[key] += val

        if self.count:
            if key not in self.counter:
                self.counter[key] = 0.
            self.counter[key] += 1.

    def update_dict(self, kv_dict):
        for k, v in kv_dict.items():
            self.update(k, v)

    def reset(self):
        self.monitor = OrderedDict()
        if self.count:
            self.counter = OrderedDict()
        if self.track_time:
            self.reset_time = time.time()

    def disp(self, reset=False):
        disp_str = ''
        if self.track_time:
            disp_str += ' time {:.2f}'.format(time.time() - self.reset_time)
        for key in sorted(self.monitor.keys()):
            if self.count:
                disp_str += ' | {}: {:.4f}'.format(key, self.monitor[key] / self.counter[key])
            else:
                disp_str += ' | {}: {}'.format(key, self.monitor[key])

        if reset:
            self.reset()

        return disp_str

def recursive_map(seq, func):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)

def recursive_to_numpy(seq):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_to_numpy(item))
        else:
            if torch.is_tensor(item):
                yield item.numpy()
            else:
                yield item

def recursive_from_numpy(seq):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_from_numpy(item))
        else:
            if isinstance(item, np.ndarray):
                yield torch.from_numpy(item)
            else:
                yield item

def log_experiment_config(config):
    disp_str = ''
    for attr in sorted(dir(config), key=lambda x: len(x)):
        if not attr.startswith('__'):
            disp_str += '{} : {}\n'.format(attr, getattr(config, attr))
    print(disp_str)

def create_log_dir(root_dir, sub_dirs=['checkpoint']):
    if os.path.exists(root_dir):
        while True:
            command = raw_input('Directory {} already exists. Do you want to delete it and proceed [Y|N]: '.format(root_dir))
            if command in ['y', 'Y']:
                shutil.rmtree(root_dir)
                break
            elif command in ['n', 'N']:
                sys.exit()
            else:
                print('Unrecognized value: {}. Please re-enter the command.'.format(command))

    os.makedirs(root_dir)
    for sub_dir in sub_dirs:
        os.makedirs("%s/%s" % (root_dir, sub_dir))

def load_vocab(vocab_dir, special_syms=None):
    if not os.path.exists(os.path.join(vocab_dir, 'vocab.pt')):
        vocab = Vocab(special_syms)
        for idx, line in enumerate(open(os.path.join(vocab_dir, 'dict.txt'), 'r')):
            vocab.add(line.strip())
        torch.save(vocab, os.path.join(vocab_dir, 'vocab.pt'))
    else:
        vocab = torch.load(os.path.join(vocab_dir, 'vocab.pt'))
    print('Vocab size: {}'.format(vocab.size()))

    return vocab

def save_checkpoint(save_path, model, optimizer, vocab, description=None):
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    checkpoint = {
            'model': model_state_dict,
            'optimizer': optim_state_dict,
            'vocab': vocab,
            'description': description
        }
    torch.save(checkpoint, save_path)

def latest_checkpoint(save_dir, prefix='chk.ep'):
    """
        Get the path for the latest checkpoint given the save dir and checkpoint prefix
    """
    fns = glob(os.path.join(save_dir, prefix) + '*')
    if len(fns) == 0:
        raise ValueError('No checkpoints with prefix "{}" find in the dir "{}"'.format(prefix, save_dir))
    latest_fn = max(fns, key=lambda fn: int(fn.split(prefix)[-1]))

    return latest_fn

def create_variable_func(cuda):
    """
        Return a modified function that turns tensor to variable which deals with:
        - checking None type
        - transfering to GPU
    """
    def variable(tensor, **kwargs):
        if tensor is None:
            return None
        if cuda:
            return Variable(tensor.cuda(), **kwargs)
        else:
            return Variable(tensor, **kwargs)

    return variable
