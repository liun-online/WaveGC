# WaveGC
This is the official implement of ICML 2025 paper 'A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition'.
This code is based on [GraphGPS](https://github.com/rampasek/GraphGPS).
![image](https://github.com/liun-online/WaveGC/blob/main/model.png)

## Python environment setup with Conda
```
# Initialize the environment
conda create -n wavegc python=3.10.13
conda activate wavegc

# Install packages about pytorch and pyg
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch_geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124
pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch

# Install other packages
conda install openbabel fsspec rdkit -c conda-forge
pip install ogb==1.3.6
pip install wandb
pip install networkx==3.2.1
pip install einops==0.7.0
pip install tqdm
pip install pandas==2.2.3
pip install tensorboardX
pip install gdown
```

## Reproduce main experiments
### a. run WaveGC on long-range datasets
Go into 'WaveGC_graph' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/{data}.yaml --repeat 4
```
Here, 'data' $\in$ {voc, pcqm, coco, pf, ps}.

### b. run WaveGC on short-range datasets
Go into 'WaveGC_node' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/{data}.yaml --repeat 10
```
Here, 'data' $\in$ {computer, corafull, cs, photo, arxiv}.

## Download PCQM dataset
cd WaveGC_graph/datasets/

gdown https://drive.google.com/u/0/uc?id=1AsDG-9WZ5b11lzHl8Ns4Qb0HxlcUOlQq && unzip pcqm4m-v2-contact.zip
