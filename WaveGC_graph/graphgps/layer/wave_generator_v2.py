import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.graphgym.config import cfg
from einops import rearrange
from graphgps.transform.transforms import path_dict

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [B, N]
        # output: [B, N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)
        
        return self.eig_w(eeig)

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class EigenEncoding(nn.Module):
    def __init__(self, hidden_dim, trans_dropout, nheads):
        super().__init__()  
        self.eig_encoder = SineEncoding(hidden_dim)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(trans_dropout)
        self.ffn_dropout = nn.Dropout(trans_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, trans_dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, eva_dense, eig_mask):
        eig = self.eig_encoder(eva_dense)  # B*N*d
        mha_eig = self.mha_norm(eig)  # B*N*d
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=eig_mask, average_attn_weights=False)
        eig = eig + self.mha_dropout(mha_eig)
        
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)  # B*N*d
        return eig
        
class Wave_generatorV2(nn.Module):
    def __init__(self, layers, pre_s, hidden_dim, nheads, num_n, num_J, trans_dropout):
        super().__init__()        
        self.num_n = num_n
        self.num_J = num_J
        self.layers = layers
                
        self.decoder_scaling = nn.Linear(hidden_dim, num_n*layers)
        self.decoder_wavelet = nn.Linear(hidden_dim, num_n*layers)
        self.decoder_scales = nn.Linear(hidden_dim, num_J*layers)
                
        self.ee = EigenEncoding(hidden_dim, trans_dropout, nheads)
        self.sigmoid = nn.Sigmoid()
        
        self.scale_scale = torch.ones(1, 1, num_J, dtype=torch.float32)
        self.scale_scale *= torch.tensor(pre_s)
        self.scale_scale = self.scale_scale.cuda()
        
        
    def gen_base(self, y, num_n, flag='scaling'):
        t_even = torch.ones(y.shape).to(y.device)
        t_odd = y
        
        base_wavelet = [t_even.unsqueeze(2)]
        base_scaling = [t_odd.unsqueeze(2)]
        for _ in range(num_n-1):
            t_even = 2*y*t_odd - t_even
            t_odd = 2*y*t_even - t_odd
            base_wavelet.append(t_even.unsqueeze(2))
            base_scaling.append(t_odd.unsqueeze(2))
        base_wavelet = torch.cat(base_wavelet, 2)
        base_scaling = torch.cat(base_scaling, 2)
        if flag == 'scaling':
            return base_scaling  # B*layers*num_n*N*1
        elif flag == 'wavelet':
            return base_wavelet  # B*layers*num_n*N*num_J
        
    def length_to_mask(self, length):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(length)
        N = length.max().item()
        mask1d  = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)

        mask2d = torch.zeros(B, N, N, device=length.device)
        for i in range(B):
            mask2d[i, :length[i], :length[i]] = 1.0

        # mask1d for key_padding_mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention
        # mask2d for edge selection from padding
        return mask1d, mask2d.bool()
    
    
    def forward(self, batch):    
        max_node = batch.eigenvector.shape[-1]
        max_length = max(batch.length)
        batch_num = batch.eigenvector.shape[0]//max_node
        
        evc_dense = batch.eigenvector.view(batch_num, max_node, max_node)[:, :max_length, :max_length]  # B*N*N
        eva_dense = batch.eigenvalue.view(batch_num, max_node)[:, :max_length]  # B*N 
        
        eig_mask, edge_idx = self.length_to_mask(batch.length)  # eig_mask B*N  edge_idx B*N*N
        
        eig_filter = self.ee(eva_dense, eig_mask)  # B*N*d
        eig_scales = eig_filter
        
        coe_scaling = self.decoder_scaling(eig_filter)  # B*N*(num_n*layers)
        coe_scaling[eig_mask] = 0.
        coe_scaling = rearrange(coe_scaling, 'a b (c d) -> a d b c', c=self.num_n, d=self.layers)  # B*layers*N*num_n
        coe_scaling = F.softmax(coe_scaling.sum(2) / (batch.length.view(-1,1,1)+1e-8), dim=-1)  # B*layers*num_n
        
        coe_wavelet = self.decoder_wavelet(eig_filter)  # B*N*(num_n*layers)
        coe_wavelet[eig_mask] = 0.
        coe_wavelet = rearrange(coe_wavelet, 'a b (c d) -> a d b c', c=self.num_n, d=self.layers)  # B*layers*N*num_n
        coe_wavelet = F.softmax(coe_wavelet.sum(2) / (batch.length.view(-1,1,1)+1e-8), dim=-1)  # B*layers*num_n
        
        coe_scales = self.decoder_scales(eig_scales)  # B*N*(num_J*layers)
        coe_scales[eig_mask] = 0.
        coe_scales = rearrange(coe_scales, 'a b (c d) -> a d b c', c=self.num_J, d=self.layers)  # B*layers*N*num_J      
        coe_scales = coe_scales.sum(2) / (batch.length.view(-1,1,1)+1e-8)  # B*layers*num_J
        coe_scales = self.sigmoid(coe_scales)*self.scale_scale  # B*layers*num_J  
        
        eva_expand = eva_dense.unsqueeze(-1).unsqueeze(1).expand(batch_num, self.layers, max_length, 1)  # B*layers*N*1 
        
        base_scaling = self.gen_base(eva_expand-1., self.num_n, 'scaling')  # B*layers*num_n*N*1
        filter_signals_wave = eva_expand * coe_scales.unsqueeze(2)  # B*layers*N*num_J
        filter_signals_wave[filter_signals_wave>2.] = 0.
        base_wavelet = self.gen_base(filter_signals_wave-1., self.num_n, 'wavelet')  # B*layers*num_n*N*num_J
        
        base_scaling = 0.5*(-base_scaling+1)
        base_wavelet = 0.5*(-base_wavelet+1)
        
        curr_scaling = coe_scaling.view(batch_num, self.layers, self.num_n, 1, 1) * base_scaling  # B*layers*num_n*N*1
        curr_scaling = curr_scaling.sum(2)  # B*layers*N*1
        
        curr_wavelet = coe_wavelet.view(batch_num, self.layers, self.num_n, 1, 1) * base_wavelet  # B*layers*num_n*N*num_J
        curr_wavelet = curr_wavelet.sum(2)  # B*layers*N*num_J
        
        filter_signals_after = torch.cat([curr_scaling, curr_wavelet], -1)  # B*layers*N*(num_J+1)
        if cfg.WaveGC.tight_frames:
            ### tight frame
            filter_signals_after = filter_signals_after / (filter_signals_after.norm(dim=-1, keepdim=True) + 1e-8)  # B*layers*N*(num_J+1)
        
        batch.evc_dense = evc_dense  # B*N*N
        batch.filter_signals_after = filter_signals_after # B*layers*N*(num_J+1)
        batch.edge_idx = edge_idx
        return batch