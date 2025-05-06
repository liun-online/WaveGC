import logging
import os
import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg


path_dict = {'COCO': 'COCOSuperpixels',
             'pcqm4m-contact': 'pcqm4m-v2-contact',
             'peptides-func': 'peptides-functional',
             'peptides-struct': 'peptides-structural',
             'Pascal': 'VOCSuperpixels'}

def cal_eig(src, dst, pad_len, num_nodes):
    if num_nodes == 1:  # some graphs have one node
        A_ = torch.tensor(1.).view(1, 1)
    else:
        A = torch.zeros([num_nodes, num_nodes], dtype=torch.float)
        A[src, dst] = 1.0
        for i in range(num_nodes):
            A[i, i] = 1.0
        deg = torch.sum(A, axis=0).squeeze() ** -0.5
        D = torch.diag(deg)
        A_ = D @ A @ D
    L = torch.eye(num_nodes) - A_
    eigenvalue, eigenvector = torch.linalg.eigh(L)
    eigenvalue = eigenvalue * 2 / (eigenvalue[-1]+1e-8)
    pad_eigenvalue = eigenvalue.new_zeros([pad_len])
    pad_eigenvalue[:num_nodes] = eigenvalue
    pad_eigenvector = eigenvector.new_zeros([pad_len, pad_len])
    pad_eigenvector[:num_nodes, :num_nodes] = eigenvector
    return pad_eigenvalue, pad_eigenvector

def wave_in_memory(dataset):
    data_list = [dataset.get(i) for i in range(len(dataset))]
    max_nodes = max([data.num_nodes for data in data_list])
    path = path_dict[cfg.wandb.project]
    eig_path = './datasets/'+path+'/eig/'
    if not os.path.exists(eig_path):
        os.makedirs(eig_path)
    for i, data in tqdm(enumerate(data_list), desc='Eigendecomposition'):
        num_nodes = data.num_nodes     
        cur_path = eig_path+'eva_'+str(i)+".pt"
        if not os.path.exists(cur_path):
            src = data.edge_index[0]
            dst = data.edge_index[1]  
            pad_eigenvalue, pad_eigenvector = cal_eig(src, dst, max_nodes, num_nodes)
            torch.save(pad_eigenvalue, eig_path+"eva_"+str(i)+".pt")
            torch.save(pad_eigenvector, eig_path+"evc_"+str(i)+".pt")
        eva = torch.load(cur_path)
        evc = torch.load(eig_path+'evc_'+str(i)+".pt")
        data.eigenvalue = eva
        data.eigenvector = evc 

        data.length = num_nodes
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

    data_list = []
    for i in tqdm(range(len(dataset)), disable=not show_progress, mininterval=10, miniters=len(dataset)//20):
        if transform_func.func.__name__ == "compute_posenc_stats":
            data_list.append(transform_func([dataset.get(i), i]))
        else:
            data_list.append(transform_func(dataset.get(i)))
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


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
