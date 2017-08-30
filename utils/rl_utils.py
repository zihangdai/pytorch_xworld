import numpy as np
import torch

def reward_to_value(rewards, gamma, baseline=None):
    values = []
    R = 0.
    for r in rewards[::-1]:
        R = r + gamma * R
        values.insert(0, R)
    if baseline:
        values = [v - baseline for v in values]

    return values

def copy_state(m1, m2):
    """
        Copy all parameters and states from m1 to m2.
        
        Often used for updating the target value network.
    """
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p2.data.copy_(p1.data)
    for key in m1._buffers.keys():
        m2._buffers[key].data.copy_(m1._buffers[key].data)
