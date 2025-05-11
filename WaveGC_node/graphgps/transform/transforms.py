import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
import os
import numpy as np
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected)
import scipy as sp
from sklearn.utils.extmath import randomized_svd

path_dict = {'computers': 'Amazon/Computers/', 
            'photo': 'Amazon/Photo/', 
            'CoraFull': 'CoraFull/cora/', 
            'cs': 'Coauthor/CS/',
            'arxiv': 'ogbn_arxiv/'}
    
def wave_in_memory(dataset):
    data_list = [dataset.get(i) for i in range(len(dataset))]
    max_nodes = max([data.num_nodes for data in data_list])
    for i, data in tqdm(enumerate(data_list), desc='Eigendecomposition'):
        num_nodes = data.num_nodes     
        eigenvalue_path = "./datasets/"+path_dict[cfg.wandb.project]+"evals.npy"
        eigenvector_path = "./datasets/"+path_dict[cfg.wandb.project]+"evects.npy"
        if os.path.exists(eigenvalue_path) and os.path.exists(eigenvector_path):
            print("\nEigenvalues and eigenvectors has been existed.")
        else:
            print("\nDecompose the adjacency matrix to get eigenvalues and eigenvectors...")
            da = dataset.get(0)
            N=da.num_nodes
            laplacian_norm_type = 'sym'
            undir_edge_index = to_undirected(da.edge_index)

            L = to_scipy_sparse_matrix(
                        *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                                    num_nodes=N)
                    )
            print(L.shape)
            if cfg.dataset.name == "ogbn-arxiv":
                sele_num = 5000
                I = sp.sparse.eye(num_nodes)
                A_sym = I - L
                U, S, _ = randomized_svd(A_sym, n_components=sele_num, random_state=42)
                eigenvalue, eigenvector = 1-S, U
            else:
                sele_num = int(cfg.WaveGC.keep_eig_ratio * num_nodes)
                eigenvalue, eigenvector = np.linalg.eigh(L.toarray())
                eigenvalue, eigenvector = eigenvalue[:sele_num], eigenvector[:, :sele_num]
            print('Finish')
            np.save(eigenvalue_path, eigenvalue)
            np.save(eigenvector_path, eigenvector)
        eigenvalue = torch.from_numpy(np.load(eigenvalue_path)).float()
        eigenvector = torch.from_numpy(np.load(eigenvector_path)).float()
        if cfg.dataset.name == "ogbn-arxiv":
            split_dict = dataset.get_idx_split()
            train_mask = torch.zeros(data.num_nodes).bool()
            val_mask = torch.zeros(data.num_nodes).bool()
            test_mask = torch.zeros(data.num_nodes).bool()
            train_mask[split_dict['train']] = True
            val_mask[split_dict['valid']] = True
            test_mask[split_dict['test']] = True
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

        data.sele_num = eigenvector.shape[1]
        data.eigenvalue = eigenvalue
        data.eigenvector = eigenvector 
                
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

def generate_splits(data, g_split):
    n_nodes = len(data.x)
    train_mask = torch.zeros(n_nodes, dtype=bool)
    valid_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    idx = torch.randperm(n_nodes)
    val_num = test_num = int(n_nodes * (1 - g_split) / 2)
    train_mask[idx[val_num + test_num:]] = True
    valid_mask[idx[:val_num]] = True
    test_mask[idx[val_num:val_num + test_num]] = True
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    return data

def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
