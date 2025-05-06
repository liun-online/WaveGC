import logging
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import (pre_transform_in_memory, wave_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'OGB':
        if name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))
    
    ## our model
    logging.info(f"Involving wavelets into dataset")
    wave_in_memory(dataset)

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    return dataset

def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgps.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset

def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgps.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgps.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset

def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
