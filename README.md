# WaveGC
This is the official implement of 'Advancing Graph Convolutional Networks via General Spectral Wavelets'.
This code is based on [GraphGPS](https://github.com/rampasek/GraphGPS).

## Python environment setup with Conda
```
conda create -n wavegc python=3.10.13
conda activate wavegc

conda install pytorch=1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.4.0 -c pyg -c conda-forge
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch

pip install ogb==1.3.6
pip install wandb
pip install networkx==3.2.1
pip install einops==0.7.0
pip install tqdm
```

## Running WaveGC on long-range datasets
Go into 'WaveGC_graph' folder, then run the following commands:
```
# GPS+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_voc.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_pcqm.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_coco.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_pf.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_ps.yaml --repeat 4 wandb.use False

# SAN+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_voc.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_pcqm.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_coco.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_pf.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_ps.yaml --repeat 4 wandb.use False

# Transformer+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_voc.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_pcqm.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_coco.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_pf.yaml --repeat 4 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_ps.yaml --repeat 4 wandb.use False
```

## Running WaveGC on short-range datasets
Go into 'WaveGC_node' folder, then run the following commands:
```
# GPS+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_computer.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_corafull.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_cs.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/gps+WaveGC_photo.yaml --repeat 10 wandb.use False

# SAN+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_computer.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_corafull.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_cs.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/san+WaveGC_photo.yaml --repeat 10 wandb.use False

# Transformer+WaveGC
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_computer.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_corafull.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_cs.yaml --repeat 10 wandb.use False
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/WaveGC/trans+WaveGC_photo.yaml --repeat 10 wandb.use False
```
