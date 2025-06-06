import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder

# @register_node_encoder('VOCNode')
# class VOCNodeEncoder(torch.nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()

#         self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
#         # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

#     def forward(self, batch):
#         batch.x = self.encoder(batch.x)

#         return batch


# @register_edge_encoder('VOCEdge')
# class VOCEdgeEncoder(torch.nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()

#         VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
#         self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
#         # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

#     def forward(self, batch):
#         batch.edge_attr = self.encoder(batch.edge_attr)
#         return batch

@register_node_encoder('VOCNode')
class VOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        node_x_mean = torch.tensor([
            4.5824501e-01, 4.3857411e-01, 4.0561178e-01, 6.7938097e-02,
            6.5604292e-02, 6.5742709e-02, 6.5212941e-01, 6.2894762e-01,
            6.0173863e-01, 2.7769071e-01, 2.6425251e-01, 2.3729359e-01,
            1.9344997e+02, 2.3472206e+02
        ])
        node_x_std = torch.tensor([
            2.5952947e-01, 2.5716761e-01, 2.7130592e-01, 5.4822665e-02,
            5.4429270e-02, 5.4474957e-02, 2.6238337e-01, 2.6600540e-01,
            2.7750680e-01, 2.5197381e-01, 2.4986187e-01, 2.6069802e-01,
            1.1768297e+02, 1.4007195e+02
        ])
        self.register_buffer('node_x_mean', node_x_mean)
        self.register_buffer('node_x_std', node_x_std)
        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)

    def forward(self, batch):
        x = batch.x - self.node_x_mean.view(1, -1)
        x /= self.node_x_std.view(1, -1)
        batch.x = self.encoder(x)
        return batch


@register_edge_encoder('VOCEdge')
class VOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        edge_x_mean = torch.tensor([0.07640745, 33.73478])
        edge_x_std = torch.tensor([0.0868775, 20.945076])
        self.register_buffer('edge_x_mean', edge_x_mean)
        self.register_buffer('edge_x_std', edge_x_std)

        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        x = batch.edge_attr - self.edge_x_mean.view(1, -1)
        x /= self.edge_x_std.view(1, -1)
        batch.edge_attr = self.encoder(x)
        return batch