import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def vis_seq_attn(ax, seq_attn, xticks, yticks, **kwargs):
    ax.set_aspect('equal')
    ax.pcolor(seq_attn, cmap=plt.cm.Blues, **kwargs)

    # ax.set_xticks(np.arange(len(xticks))+0.5)
    # ax.set_xticklabels(xticks, fontsize=12, rotation='30')
    # ax.set_yticks(np.arange(len(yticks))+0.5)
    # ax.set_yticklabels(yticks, fontsize=12)

    ax.set_xticks(np.arange(len(xticks))+0.5, minor=True)
    ax.set_xticklabels(xticks, fontsize=12, minor=True, rotation='60')
    ax.set_yticks(np.arange(len(yticks))+0.5, minor=True)
    ax.set_yticklabels(yticks, fontsize=12, minor=True)

    ax.set_xticks(np.arange(len(xticks)), minor=False)
    ax.set_xticklabels([], minor=False)
    ax.set_yticks(np.arange(len(yticks)), minor=False)
    ax.set_yticklabels([], minor=False)

    ax.grid(True, which='major', linestyle='-', color='white')

def vis_scale_image(image, scale):
    if isinstance(scale, tuple):
        s_w, s_h = scale
    elif isinstance(scale, int):
        s_w, s_h = scale, scale
    else:
        raise ValueError('scale should be either an int or a tuple, but got {}'.format(scale))

    B, C, W, H = image.size()
    scaled_image = image.view(B, C, W, 1, H, 1).repeat(1, 1, 1, s_w, 1, s_h).view(B, C, W*s_w, H*s_h)

    return scaled_image

opencv_inv_idx = torch.arange(2,-1,-1).long()
def opencv_to_rgb(image):
    return image.index_select(1, opencv_inv_idx)

def clear_axes(axes):
    for ax in axes:
        if isinstance(ax, np.ndarray):
            clear_axes(ax)
        else:
            ax.clear()