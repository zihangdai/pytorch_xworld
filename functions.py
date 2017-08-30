import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

if torch.__version__ == '0.1.12_2':
    kwargs = {}
else:
    kwargs = {keepdim:True}

#################### Math ####################
def entropy_from_prob(prob, dim=-1):
    return -torch.sum(torch.log(1e-6 + prob) * prob, dim=dim)

def log_sum_exp(logits):
    max_logits = logits.max(1, **kwargs)[0]

    return torch.log(1e-6 + (logits - max_logits.expand_as(logits)).exp().sum(1)) + max_logits.squeeze(1)

#################### Attention ####################
def temporal_weigthed_avg(context, attn, batch_first=False):
    """
        context : [batch x seqlen x dim] if batch_first else [seqlen x batch x dim]
        attn    : [batch x seqlen]
    """
    if not batch_first:
        context = context.transpose(0, 1)                           # batch x seqlen x dim
    avg_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)  # batch x dim

    return avg_context

def spatial_weighted_avg(context, attn):
    """
        context : [batch x dim x H x W]
        attn    : [batch x  1  x H x W]
    """
    
    avg_context = (context * attn.expand_as(context)) \
                  .sum(3, **kwargs).sum(2, **kwargs)  \
                  .squeeze(3).squeeze(2)                 # batch x dim

    return avg_context

def dotprod_attention(context, query, mask=None):
    """
        context : [seqlen x batch x dim]
        query   : [   1   x batch x dim]
        mask.   : [seqlen x batch]
    """
    context = context.transpose(0, 1)             # batch x seqlen x dim
    query = query.squeeze(0).unsqueeze(2)         # batch x dim    x 1
    attn = torch.bmm(context, query).squeeze(2)   # batch x seqlen
    if mask is not None:
        attn.data.masked_fill_(mask.t(), -float('inf'))
    attn_prob = F.softmax(attn)
    
    return attn_prob

def cosine_attention(context, query, mask=None):
    """
        context : [seqlen x batch x dim]
        query   : [   1   x batch x dim]
        mask.   : [seqlen x batch]
    """
    
    context = context / (1e-6 + context.norm(p=2, dim=2, **kwargs)).expand_as(context)
    query = query / (1e-6 + query.norm(p=2, dim=2, **kwargs).expand_as(query))

    return dotprod_attention(context, query, mask=None)

#################### RNN ####################
def bihidden_to_unihidden(h):
    """
        Concat the final hidden states (fwd and bwd) of a bidirectional RNN 
        to create a hidden state for a unidirecitonal RNN
    """
    return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                  .transpose(1, 2).contiguous() \
                  .view(h.size(0) // 2, h.size(1), h.size(2) * 2)

def create_rnn_step(rnn):
    """
        Given an RNN, create a corresponding one-step function
    """
    def step_func(input, hid=None):
        rnn_step = F._functions.rnn.AutogradRNN(
            mode=rnn.mode, input_size=rnn.input_size, hidden_size=rnn.hidden_size, 
            num_layers=rnn.num_layers, dropout=rnn.dropout,
            train=rnn.training, bidirectional=rnn.bidirectional)
        return rnn_step(input, rnn.all_weights, hid)

    return step_func

def check_decreasing(lengths):
    """
        Check whether the lengths tensor are in descreasing order.
        - If true, return None
        - Else, return a decreasing lens with two mappings

        This is used for variable length RNN
    """
    lens, order = torch.sort(lengths, 0, True) 
    if torch.ne(lens, lengths).sum() == 0:
        return None
    else:
        _, rev_order = torch.sort(order)

        return lens, Variable(order), Variable(rev_order)

def rnn_varlen(rnn, seq, emb, hidden=None):
    """
        Process a sequential embedding (`emb`) using a variable length RNN,
        where the LongTensor (`seq`) is passed in to compute the mask and lengths
    """
    padmask = seq.data.eq(0)
    lengths = seq.data.ne(0).sum(0, **kwargs).squeeze(0)
    check_res = check_decreasing(lengths)

    if check_res is None:
        packed_emb = rnn_utils.pack_padded_sequence(emb, lengths.tolist())
        packed_out, hidden_final = rnn(packed_emb, hidden)
        outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
    else:
        lens, order, rev_order = check_res
        packed_emb = rnn_utils.pack_padded_sequence(emb.index_select(1, order), lens.tolist())
        packed_out, hidden_final = rnn(packed_emb, hidden)
        outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
        outputs = outputs.index_select(1, rev_order)
        hidden_final = hidden_final.index_select(1, rev_order)

    if padmask.size(0) > outputs.size(0):
        padmask = padmask.narrow(0, 0, outputs.size(0))

    return outputs, hidden_final, padmask, lengths

#################### Misc ####################
def create_centered_map(batch_size, grid_size):
    centered_map = Variable(torch.zeros(batch_size, 1, grid_size, grid_size))
    centered_map.data[:,:,grid_size//2, grid_size//2].fill_(1.)
    
    return centered_map

def conv_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    out_size = (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    return out_size

def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def param_norm(parameters, norm_type=2):
    parameters = list(parameters)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def linear_schedule(init_val, final_val, progress):
    return init_val + min(1., progress) * (final_val - init_val)
