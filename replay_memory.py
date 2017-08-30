import numpy as np
import torch
from collections import namedtuple
import random

from segment_tree import SumSegmentTree, MinSegmentTree

class Experience(object):
    def __init__(self, state, action, reward, done):
        """
            state  :  tuple : image and command
            action : tensor : action taken
            reward :  float : immediate reward after taking the action
            done   :   bool : whether the episode ends after taking the action
        """
        self.state  = state
        self.action = action
        self.reward = reward
        self.done   = done

class ReplayMemory(object):
    def __init__(self, capacity, pack_func=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.pack_func = pack_func

    def push(self, transition):
        """Saves a transition."""
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Packs a batch"""
        batch = random.sample(self.memory, batch_size)
        if self.pack_func is not None:
            batch = self.pack_func(batch)
        return batch

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, pack_func=None):
        self.capacity = capacity

        assert alpha > 0
        self._alpha = alpha

        self.pack_func = pack_func

        self.memory = []
        self.position = 0

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, transition):
        """Saves a transition."""
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self._it_sum[self.position] = self._max_priority ** self._alpha
        self._it_min[self.position] = self._max_priority ** self._alpha

        self.position = (self.position + 1) % self.capacity

    def _sample_proportional(self, batch_size):
        indices = []
        while len(indices) < batch_size:
            mass = random.random() * self._it_sum.sum(0, len(self.memory) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx not in indices:
                indices.append(idx)
        return indices

    def sample(self, batch_size, beta):
        """Packs a batch"""
        indices = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        batch = []
        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
            batch.append(self.memory[idx])
        weights = torch.Tensor(weights)

        if self.pack_func is not None:
            batch = self.pack_func(batch)
        return batch, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index indices[i] in buffer
        to priorities[i].

        Parameters
        ----------
        indices: [int]
            List of indices of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled indices denoted by
            variable `indices`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0, priority
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self.memory)


def pack_varlen_seqs(seqs):
    max_len = max(seq.size(0) for seq in seqs)
    out = seqs[0].new(max_len, len(seqs)).fill_(0)
    for i in range(len(seqs)):
        seqlen = seqs[i].size(0)
        out[:seqlen,i].copy_(seqs[i][:,0])

    return out

def pack_batch_nav(batch):
    def pack_state(state):
        image, command = state
        image = torch.cat(image, dim=0)
        command = pack_varlen_seqs(command)

        return image, command

    curr_state, action, next_state, reward = zip(*batch)

    curr_state = pack_state(zip(*curr_state))
    next_state = pack_state(zip(*next_state))
    action = torch.LongTensor(action)
    reward = torch.Tensor(reward)

    return curr_state, action, next_state, reward

def pack_batch_rec(batch):
    image, question, answer = zip(*batch)
    image = torch.cat(image, dim=0)
    answer = torch.cat(answer, dim=0)
    question = pack_varlen_seqs(question)

    return image, question, answer
