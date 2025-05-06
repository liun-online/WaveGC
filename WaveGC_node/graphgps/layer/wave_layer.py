import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.data import Batch
import torch_geometric.nn as pygnn

from torch_geometric.graphgym.config import cfg
from graphgps.layer.gatedgcn_layer import GatedGCNLayer

class Wave_Conv_arxiv(nn.Module):
    def __init__(self, dim_h, act, dropout, num_J):
        super().__init__()
        self.dim_h = dim_h
        self.single_out = self.dim_h//(num_J+1)
        self.single_out_last = self.dim_h - num_J*self.single_out
        self.activation = register.act_dict[act]()
        self.drop = nn.Dropout(dropout)
        self.t_number = num_J + 1

        self.lin = nn.ModuleList([nn.Linear(dim_h, self.single_out) for _ in range(self.t_number-1)])
        self.lin.append(nn.Linear(dim_h, self.single_out_last))
        self.fusion = nn.Linear(dim_h, dim_h)
    
    def forward(self, x, batch):
        comb = []
        evc_dense = batch.eigenvector.unsqueeze(0)
        evc_dense_t = evc_dense.transpose(1, 2)
        curr_signals = batch.filter_signals_after # 1*sele_num*(n_scales+1)
        curr_signals = curr_signals.transpose(2, 1)
        for t in range(self.t_number):
            curr_sig = curr_signals[:, t, :]
            h = self.lin[t](x)
            h = evc_dense_t @ h
            h = curr_sig.unsqueeze(-1) * h
            h = self.drop(self.activation(evc_dense @ h))

            h = evc_dense_t @ h
            h = curr_sig.unsqueeze(-1) * h
            h = evc_dense @ h
            comb.append(h)
        comb = torch.cat(comb, -1)
        final_emb = self.drop(self.activation(self.fusion(comb)))[0]
        return final_emb
    
class Wave_Conv(nn.Module):
    def __init__(self, dim_h, act, dropout, num_J):
        super().__init__()
        self.dim_h = dim_h
        self.single_out = self.dim_h//(num_J+1)
        self.single_out_last = self.dim_h - num_J*self.single_out
        self.activation = register.act_dict[act]()
        self.drop = nn.Dropout(dropout)
        self.t_number = num_J + 1
        self.wavelet_conv_2 = nn.ModuleList([pygnn.SimpleConv()for _ in range(self.t_number)])
        self.wavelet_conv_1 = nn.ModuleList([pygnn.GCNConv(dim_h, self.single_out, normalize=False)  for _ in range(self.t_number-1)])
        self.wavelet_conv_1.append(pygnn.GCNConv(dim_h, self.single_out_last, normalize=False))
        self.fusion = nn.Linear(dim_h, dim_h)
    
    def forward(self, x, batch):
        comb = []
        evc_dense = batch.eigenvector.unsqueeze(0)
        evc_dense_t = evc_dense.transpose(1, 2)
        curr_signals = batch.filter_signals_after # 1*sele_num*(n_scales+1)
        curr_signals = curr_signals.transpose(2, 1)  # 1*(n_scales+1)*sele_num
        for t in range(self.t_number):
            filters = evc_dense @ (curr_signals[:, t, :].unsqueeze(-1) * evc_dense_t)
            filters = F.normalize(filters)
            sele_filters = torch.abs(filters) > cfg.WaveGC.keep_thre
            
            attr = filters[sele_filters].to(torch.float32)
            index = sele_filters[0].nonzero().t()
            h = self.drop(self.activation(self.wavelet_conv_1[t](x, index, attr)))
            h = self.wavelet_conv_2[t](h, index, attr)
            comb.append(h)
        comb = torch.cat(comb, -1)
        final_emb = self.drop(self.activation(self.fusion(comb)))
        return final_emb

class WaveLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 layer_type, num_heads, num_J, act='relu', equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True, log_attn_weights=False, wave_drop=0.):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )
        
        try:
            local_gnn_type, global_model_type = layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {layer_type}")
            
        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'WaveGC':
            if cfg.dataset.name == "ogbn-arxiv":
                self.self_attn = Wave_Conv_arxiv(dim_h, act, wave_drop, num_J)
            else:
                self.self_attn = Wave_Conv(dim_h, act, wave_drop, num_J)
            self.attn_flag = 'wavelet'
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                    batch.edge_index,
                                                    batch.edge_attr,
                                                    batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                    batch.edge_index,
                                                    batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.attn_flag == 'wavelet':
                h_attn = self.self_attn(h, batch)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
