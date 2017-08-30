
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils 

from functions import *
from utils import *
import random

USE_VARLEN_RNN = True

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class Agent(nn.Module):
    def __init__(self, config, vocab_size):
        super(Agent, self).__init__()

        self.config = config
        self.alpha = config.alpha

        # WordEmbed maintains the embedding matrix which is also shared as the softmax matrix in recognition task
        self.WordEmbed = nn.Embedding(vocab_size, config.word_embed_dim, padding_idx=0)

        # FuncEmbed_MLP and SyntEmbed_MLP project word embeddings into functional embeddings and syntactic embeddings respectively
        self.FuncEmbed_MLP = nn.Sequential(
            nn.Linear(config.word_embed_dim, config.word_embed_dim//2), nn.Tanh(), 
            nn.Linear(config.word_embed_dim//2, config.func_embed_dim), nn.Tanh()
        )
        self.SyntEmbed_MLP = nn.Sequential(
            nn.Linear(config.word_embed_dim, config.word_embed_dim//2), nn.Tanh(), 
            nn.Linear(config.word_embed_dim//2, config.synt_embed_dim), nn.Tanh()
        )

        # Controller_RNN takes the syntactic embedding as input and encode it
        self.Controller_RNN = nn.GRU(config.synt_embed_dim, config.lang_hidden_dim, bidirectional=True)
        
        # Context_MLP projects the concatenation of syntactic embedding and controller output into context vectors
        self.Context_MLP = nn.Sequential(nn.Linear(config.synt_embed_dim+2*config.lang_hidden_dim, config.lang_hidden_dim), nn.Tanh())
        
        # Booting_MLP projects the final state of the controller into the initial state of the programmer RNN
        self.Booting_MLP = nn.Sequential(nn.Linear(2*config.lang_hidden_dim, config.lang_hidden_dim), nn.Tanh())
        
        # Encoder_MLP projects the controller output into the context key used in computing word attention
        self.Encoder_MLP = nn.Sequential(nn.Linear(config.lang_hidden_dim, config.lang_hidden_dim), nn.Tanh())

        # Programmer_RNN maintains the recurrent internal state of the Programmer
        self.Programmer_RNN = nn.GRU(config.lang_hidden_dim, config.lang_hidden_dim)

        # Sigma_MLP computes the ratio for mixing the grid attention map across programming steps
        # self.Sigma_MLP = nn.Sequential(nn.Linear(2*config.lang_hidden_dim, 1, bias=False), nn.Sigmoid())
        self.Sigma_MLP = nn.Sequential(nn.Linear(2*config.lang_hidden_dim, 2, bias=False), nn.Softmax())

        # Mask_MLP takes functional embeddings as input to produce a mask
        self.Mask_MLP = nn.Sequential(
            nn.Linear(config.func_embed_dim, config.func_embed_dim), nn.Tanh(), 
            nn.Linear(config.func_embed_dim, config.word_embed_dim), nn.Sigmoid()
        )

        # Question_RNN encodes the question in recognition task
        self.Question_RNN = nn.GRU(config.func_embed_dim, config.func_embed_dim)

        # Visual_CNN extracts visual information from the raw image input
        self.Visual_CNN = nn.Sequential(
            nn.Conv2d(                        3, config.vis_num_filters[0], config.vis_filter_sizes[0], stride=config.vis_strides[0]), nn.ReLU(), 
            nn.Conv2d(config.vis_num_filters[0], config.vis_num_filters[1], config.vis_filter_sizes[1], stride=config.vis_strides[1]), nn.ReLU(), 
            nn.Conv2d(config.vis_num_filters[1], config.vis_num_filters[2], config.vis_filter_sizes[2], stride=config.vis_strides[2]), nn.ReLU(), 
            nn.Conv2d(config.vis_num_filters[2], config.vis_num_filters[3], config.vis_filter_sizes[3], stride=config.vis_strides[3]), nn.ReLU()
        )
        
        # feat_map_spatial as a trainable parameter represents the spatial/directional information
        self.feat_map_spatial = nn.Parameter(torch.Tensor(config.feat_spatial_dim, config.grid_size, config.grid_size))

        # Env_CNN transforms visual feature map into an environment map (with only one channel) using 1x1 convolution
        self.Env_CNN = nn.Sequential(nn.Conv2d(in_channels=config.feat_visual_dim, out_channels=1, kernel_size=1), nn.Sigmoid())

        # State_Net transforms the concatenation of the environment map and the grid attention map into the state representation of the MDP
        self.State_Net = nn.Sequential(
            nn.Conv2d(                        2, config.act_num_filters[0], config.act_filter_size, padding=config.act_padding), nn.ReLU(),
            nn.Conv2d(config.act_num_filters[0], config.act_num_filters[1], config.act_filter_size, padding=config.act_padding), # NOTE: No activition here 
            Expression(lambda x: x.view(x.size(0), -1)), # Flatten 4D tensor to 2D tensor
            nn.Linear(config.act_num_filters[1]*config.grid_size*config.grid_size, config.act_hidden_dim), nn.ReLU(),
            nn.Linear(config.act_hidden_dim, config.act_hidden_dim), nn.ReLU()
        )

        # Action_Net outputs the policy action probability given the state representation
        self.Action_Net = nn.Sequential(nn.Linear(config.act_hidden_dim, config.num_actions), nn.Softmax())

        # Value_Net outputs the predicted value given the current state representation
        self.Value_Net = nn.Sequential(
            nn.Linear(config.act_hidden_dim, config.act_hidden_dim//2), nn.ReLU(), 
            nn.Linear(config.act_hidden_dim//2, 1)
        )

        # ===== Additional working memory
        self.inv_idx = torch.arange(config.grid_size-1,-1,-1).long()
        # self.inv_idx = Variable(torch.arange(config.grid_size-1,-1,-1).long())
        if config.cuda: self.inv_idx = self.inv_idx.cuda()

        # Parameter intialization
        self._reset_parameters()

    def _reset_parameters(self):
        def normal_init(name, std):
            def init_func(m):
                if name in m.__class__.__name__:
                    m.weight.data.normal_(mean=0., std=std)
                    if m.bias is not None:
                        m.bias.data.fill_(0.)
            return init_func

        def gru_init(m):
            if 'GRU' in m.__class__.__name__:
                for l in range(m.num_layers):
                    for name in ['ih', 'hh']:
                        weight = getattr(m, 'weight_{}_l{}'.format(name, l))
                        w_size = weight.data.numel()
                        weight.data.normal_(mean=0., std=1.0)

        def smart_init(name):
            def init_func(m):
                if name in m.__class__.__name__:
                    w_size = m.weight.data.numel()
                    m.weight.data.normal_(mean=0., std=np.sqrt(2./w_size))
                    if m.bias is not None:
                        m.bias.data.fill_(0.)
            return init_func

        def zero_init(m):
            if 'Linear' in m.__class__.__name__: 
                m.weight.data.fill_(0.)
                if m.bias is not None:
                    m.bias.data.fill_(0.)

        self.WordEmbed.weight.data.normal_(mean=0., std=1.)
        self.feat_map_spatial.data.zero_()

        self.apply(normal_init('Conv2d', std=0.1))

    def get_grid_attention(self, feat_map, text):
        config = self.config

        ##### Language
        ## Embedding
        word_embed = self.WordEmbed(text)
        func_embed = self.FuncEmbed_MLP(word_embed.view(-1, config.word_embed_dim)).view(word_embed.size(0), -1, config.func_embed_dim)
        synt_embed = self.SyntEmbed_MLP(word_embed.view(-1, config.word_embed_dim)).view(word_embed.size(0), -1, config.synt_embed_dim)

        ## Controller
        if USE_VARLEN_RNN:
            encoded_all, encoded_final, padmask, lengths = rnn_varlen(self.Controller_RNN, text, synt_embed)
        else:
            encoded_all, encoded_final = self.Controller_RNN(synt_embed)
        context_vec = self.Context_MLP(torch.cat([encoded_all, synt_embed], dim=2).view(-1, config.synt_embed_dim+2*config.lang_hidden_dim)).view(encoded_all.size(0), -1, config.lang_hidden_dim)

        encoded_final = bihidden_to_unihidden(encoded_final)
        programmer_boot = self.Booting_MLP(encoded_final.view(-1, 2*config.lang_hidden_dim)).view(encoded_final.size(0), -1, config.lang_hidden_dim)

        ## Programmer
        # initial states for the programmer recurrence
        cached_attn = create_centered_map(feat_map.size(0), config.grid_size)
        if config.cuda: cached_attn = cached_attn.cuda()

        # obtain the "unit norm" context key used for attention; unit norm is used because we want cosine similarity
        context_key = self.Encoder_MLP(context_vec.view(-1, config.lang_hidden_dim)).view(context_vec.size(0), -1, config.lang_hidden_dim)

        # core programming recurrence
        grid_attns, cached_attns, heatmaps, seq_attns, masks, sigmas = [], [], [], [], [], []
        hiddens = [programmer_boot]
        program_rnn_step = create_rnn_step(self.Programmer_RNN)
        for step in range(config.program_steps):
            if USE_VARLEN_RNN:
                attn_prob = cosine_attention(context_key, hiddens[-1], padmask)
            else:
                attn_prob = cosine_attention(context_key, hiddens[-1])
            avg_context_vec = temporal_weigthed_avg(context_vec, attn_prob)
            avg_word_embed  = temporal_weigthed_avg(word_embed, attn_prob)
            avg_func_embed  = temporal_weigthed_avg(func_embed, attn_prob)

            # update programmer RNN hidden state using average context vectors
            _, new_hidden = program_rnn_step(avg_context_vec.unsqueeze(0), hiddens[-1])
            hiddens.append(new_hidden)

            # compute masked avg_word_embed
            mask = self.Mask_MLP(avg_func_embed)
            masked_word_embed = avg_word_embed * mask

            # create a spatially normalized heatmap which will be used as "filter" to convolve the grid_attn_map and achieve 2D translation
            # - Dimension: masked_word_embed [batch x dim], feat_map [batch x dim x grid x grid]
            heatmap = F.softmax(torch.bmm(masked_word_embed.unsqueeze(1), feat_map.view(feat_map.size(0), feat_map.size(1), -1)).squeeze(1))
            heatmap = heatmap.view(feat_map.size(0), 1, feat_map.size(2), feat_map.size(3))

            # rotation
            inv_idx = Variable(self.inv_idx)
            trans_filter = heatmap.index_select(2, inv_idx).index_select(3, inv_idx)

            # mapping to the terms in the paper: trans_filter is the `o(a_s'')`, cached_attn is `a_{s-1}'`, grid_attn is the `a_s`
            # a_s = Conv2d ( o(a_s'') , a_{s-1}' )
            grid_attn = nn.functional.conv2d(cached_attn.permute(1, 0, 2, 3), trans_filter, padding=config.grid_size//2, groups=trans_filter.size(0)).permute(1, 0, 2, 3)

            # soft update cached_attn
            sigma = self.Sigma_MLP(torch.cat([new_hidden.squeeze(0), avg_context_vec], dim=1))

            prev_sigma = sigma.narrow(1, 0, 1)
            curr_sigma = sigma.narrow(1, 1, 1)
            prev_sigma_expand = prev_sigma.unsqueeze(2).unsqueeze(3).expand_as(cached_attn)
            curr_sigma_expand = curr_sigma.unsqueeze(2).unsqueeze(3).expand_as(grid_attn)
            cached_attn = prev_sigma_expand * cached_attn + curr_sigma_expand * grid_attn
 
            # quantities to monitor
            cached_attns.append(cached_attn)
            grid_attns.append(grid_attn)
            seq_attns.append(attn_prob)
            heatmaps.append(heatmap)
            sigmas.append(prev_sigma)
            masks.append(mask)

        return grid_attn, func_embed, {'cached_attns':cached_attns, 'grid_attns':grid_attns, 'heatmaps':heatmaps, 'seq_attns':seq_attns, 'masks':masks, 'sigmas':sigmas}

    def get_action(self, state):
        action_prob = self.Action_Net(state)
        if self.training:
            if random.random() > self.alpha:
                action = action_prob.multinomial(1)
            else:
                action = Variable(torch.LongTensor(action_prob.size(0), 1).random_(4))
                if self.config.cuda: action = action.cuda()
        else:
            action = action_prob.multinomial(1)

        return action, action_prob

    def get_value(self, state):
        value = self.Value_Net(state)
        return value

    def forward(self, image, command=None, question=None, act_only=False, val_only=False):
        assert not (act_only and val_only), 'act_only and val_only cannot be both True'
        
        config = self.config

        ##### Perception
        feat_map_visual = self.Visual_CNN(image)
        feat_map = torch.cat([feat_map_visual, self.feat_map_spatial.unsqueeze(0).expand_as(feat_map_visual)], dim=1)
        # feat_map = torch.cat([feat_map_visual, expand_as(self.feat_map_spatial.unsqueeze(0), feat_map_visual)], dim=1)

        ##### Action and Value
        if command is not None:
            grid_attn_act, _, nav_info = self.get_grid_attention(feat_map, command)
            env_map = self.Env_CNN(feat_map_visual)
            state = self.State_Net(torch.cat([env_map, grid_attn_act], dim=1))

            nav_info['env_map'] = env_map

            if act_only:
                action, action_prob = self.get_action(state)
                return action, nav_info
            elif val_only:
                value = self.get_value(state)
                return value, nav_info
            else:
                action, action_prob = self.get_action(state)
                value = self.get_value(state)
                return action, action_prob, value, nav_info

        ##### Recognition
        if question is not None:
            rec_grid_attn, rec_func_embed, rec_info = self.get_grid_attention(feat_map, question)
            rec_feat = spatial_weighted_avg(feat_map, rec_grid_attn)

            if USE_VARLEN_RNN:
                rec_hidden_all, rec_hidden_final, rec_padmask, rec_lengths = rnn_varlen(self.Question_RNN, question, rec_func_embed)
                avg_rec_func_embed = rec_hidden_all.sum(0, **kwargs).squeeze(0)
                avg_rec_func_embed = avg_rec_func_embed / Variable(rec_lengths.float()).unsqueeze(1).expand_as(avg_rec_func_embed)
                # avg_rec_func_embed = avg_rec_func_embed / expand_as(Variable(rec_lengths.float()).unsqueeze(1), avg_rec_func_embed)
            else:
                avg_rec_func_embed = self.Question_RNN(rec_func_embed)[0].mean(0).squeeze(0)

            rec_mask = self.Mask_MLP(avg_rec_func_embed)
            rec_logit = torch.mm(rec_feat * rec_mask, self.WordEmbed.weight.t())

            # monitor information
            rec_info['rec_mask'] = rec_mask

            return rec_logit, rec_info
