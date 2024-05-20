# WaveGC
This is the official implement of 'Advancing Graph Convolutional Networks via General Spectral Wavelets'.
This code is based on [GraphGPS](https://github.com/rampasek/GraphGPS).
![image](https://github.com/liun-online/WaveGC/blob/main/WaveGC_model.png)

## Python environment setup with Conda
```
# Initialize the environment
conda create -n wavegc python=3.10.13
conda activate wavegc

# Install packages about pytorch and pyg
conda install pytorch=1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.4.0 -c pyg -c conda-forge
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch

# Install other packages
conda install openbabel fsspec rdkit -c conda-forge
pip install ogb==1.3.6
pip install wandb
pip install networkx==3.2.1
pip install einops==0.7.0
pip install tqdm
pip install pandas==1.5.3
pip install tensorboardX
pip install gdown
```

## Reproduce main experiments
### a. run WaveGC on long-range datasets
Go into 'WaveGC_graph' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/{model}+WaveGC_{data}.yaml --repeat 4 wandb.use False
```
Here, 'model' $\in$ {gps, san, trans}, 'data' $\in$ {voc, pcqm, coco, pf, ps}. E.g.,
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_voc.yaml --repeat 4 wandb.use False
```

### b. run WaveGC on short-range datasets
Go into 'WaveGC_node' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/{model}+WaveGC_{data}.yaml --repeat 10 wandb.use False
```
Here, 'model' $\in$ {gps, san, trans}, 'data' $\in$ {computer, corafull, cs, photo}. E.g.,
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_computer.yaml --repeat 10 wandb.use False
```

### c. run WaveGC on ogbn-arxiv
Go into 'WaveGC_arxiv' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/{model}+WaveGC_arxiv.yaml --repeat 10 wandb.use False
```
Here, 'model' $\in$ {gps, san, trans}. E.g.,
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_arxiv.yaml --repeat 10 wandb.use False
```
### d. run WaveGC on heterophily datasets
Go into 'WaveGC_hete' folder, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/{model}+WaveGC_{data}.yaml --repeat 10 wandb.use False
```
Here, 'model' $\in$ {gps, san, trans}, 'data' $\in$ {actor, mine, tolo}. E.g.,
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_actor.yaml --repeat 10 wandb.use False
```

## Download PCQM dataset
cd WaveGC_graph/datasets/

gdown https://drive.google.com/u/0/uc?id=1AsDG-9WZ5b11lzHl8Ns4Qb0HxlcUOlQq && unzip pcqm4m-v2-contact.zip
