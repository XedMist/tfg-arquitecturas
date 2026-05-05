"""
train_coco.py — Script de entrenamiento para detección con MMDetection.

Equivalente a tools/train.py de MMDetection, pero integrado con los
backbones MetaFormer de este proyecto.

Uso:
    python src/detection/tools/train_coco.py \\
        src/configs/detection/faster_rcnn_gated_cnn_mamba_fpn_1x_coco.py
"""

import argparse
import os
import sys
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic mixed precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    # ── Registrar componentes del proyecto ────────────────────────────────────
    # Asegurar que src/ está en el path para importar detection.backbones
    _src = Path(__file__).resolve().parents[2]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    
    # Importar para registrar MetaFormerBackbone en el registry de MMDetection
    import detection  # noqa: F401

    # Cargar config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Work dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Usar nombre de la config como default
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # AMP
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('AMP ya está activado en la config.')
        else:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # Auto scale LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('No se encontró configuración auto_scale_lr en el archivo.')

    # Construir runner e iniciar
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
