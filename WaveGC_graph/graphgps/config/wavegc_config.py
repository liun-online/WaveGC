from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_WaveGC')
def set_cfg_lam(cfg):
    cfg.WaveGC = CN()
    cfg.WaveGC.nheads = 8
    cfg.WaveGC.trans_dropout = 0.1
    cfg.WaveGC.drop = 0.5
    cfg.WaveGC.num_n = 5
    cfg.WaveGC.num_J = 3
    cfg.WaveGC.pre_s = [10.0, 10.0, 10.0]
    cfg.WaveGC.tight_frames = True
    cfg.WaveGC.weight_share = False
    cfg.WaveGC.trans_use = False
    cfg.WaveGC.normalize = True
