from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_WaveGC')
def set_cfg_WaveGC(cfg):
    cfg.WaveGC = CN()
    cfg.WaveGC.nheads = 8
    cfg.WaveGC.trans_dropout = 0.1
    cfg.WaveGC.drop = 0.5
    cfg.WaveGC.num_n = 5
    cfg.WaveGC.num_J = 3
    cfg.WaveGC.pre_s = [10.0, 10.0, 10.0]
    cfg.WaveGC.tight_frames = True
    cfg.WaveGC.keep_eig_ratio = 0.05
    cfg.WaveGC.keep_thre = 1e-3