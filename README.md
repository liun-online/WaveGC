# WaveGC
This is the official implement of 'Advancing Graph Convolutional Networks via General Spectral Wavelets'.
This code is based on [GraphGPS](https://github.com/rampasek/GraphGPS).

## Environment Settings
```
python==3.10.13
networkx==3.2.1
numpy==1.26.0
scikit-learn=1.3.0
scipy==1.11.3
torch==1.13.1
torch_geometric==2.4.0
ogb==1.3.6
einops==0.7.0
tqdm==4.66.1
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
