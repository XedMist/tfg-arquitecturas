"""
train.py
========
Script de entrenamiento de Mask2Former con tu MetaformerBackbone.

Uso básico:
    # Una GPU (desarrollo / prueba rápida)
    python train.py --config-file configs/maskformer2_metaformer_coco_panoptic.yaml

    # Varias GPUs (recomendado para COCO completo)
    python train.py --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
                    --num-gpus 4

    # Reanudar un entrenamiento interrumpido
    python train.py --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
                    --num-gpus 4 --resume

    # Sobreescribir parámetros sin tocar el yaml
    python train.py --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
                    SOLVER.BASE_LR 0.00005 OUTPUT_DIR ./output/experimento_lr_bajo
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Mask2Former
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

# Tu backbone
from metaformer_backbone_d2 import MetaformerBackbone, add_metaformer_config  # noqa: F401

logger = logging.getLogger("detectron2")


# ---------------------------------------------------------------------------
# Trainer personalizado
# ---------------------------------------------------------------------------
class MetaformerTrainer(DefaultTrainer):
    """
    Extiende DefaultTrainer para:
      - Usar los dataset mappers correctos de Mask2Former según la tarea
      - Construir el evaluador adecuado (panóptico / instancias / semántico)
      - Aplicar el multiplicador de LR al backbone (fine-tuning diferencial)
    """

    # ------------------------------------------------------------------
    # Dataset mapper: cómo se cargan y aumentan las imágenes
    # ------------------------------------------------------------------
    @classmethod
    def build_train_loader(cls, cfg):
        mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        if mapper_name == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, is_train=True)
        elif mapper_name == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=True)
        elif mapper_name == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)
        elif mapper_name == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, is_train=True)
        elif mapper_name == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        else:
            raise ValueError(f"Dataset mapper desconocido: {mapper_name}")

        from detectron2.data import build_detection_train_loader
        return build_detection_train_loader(cfg, mapper=mapper)

    # ------------------------------------------------------------------
    # Evaluadores: calculan PQ, AP y mIoU automáticamente
    # ------------------------------------------------------------------
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluators = []
        meta = MetadataCatalog.get(dataset_name)

        # Segmentación semántica → mIoU
        if cfg.MODEL.MASK_ON is False:
            evaluators.append(
                SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder)
            )

        # Segmentación de instancias → mask AP
        if cfg.MODEL.MASK_ON:
            evaluators.append(
                COCOEvaluator(dataset_name, output_dir=output_folder)
            )

        # Segmentación panóptica → PQ
        if hasattr(meta, "panoptic_root"):
            evaluators.append(
                COCOPanopticEvaluator(dataset_name, output_folder)
            )

        # Fallback por si ninguno aplica
        if not evaluators:
            evaluators.append(
                COCOEvaluator(dataset_name, output_dir=output_folder)
            )

        return DatasetEvaluators(evaluators)

    # ------------------------------------------------------------------
    # Optimizador con LR diferencial backbone vs. cabezas
    # ------------------------------------------------------------------
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Aplica BACKBONE_LR_MULTIPLIER al stem y stages del backbone.
        El resto del modelo (Pixel Decoder + Transformer Decoder) usa
        el BASE_LR completo.
        """
        backbone_multiplier = cfg.SOLVER.get("BACKBONE_LR_MULTIPLIER", 0.1)
        base_lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM

        # Identificamos parámetros del backbone
        backbone_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = "backbone" in name
            no_decay = any(
                nd in name for nd in ["bias", "norm.weight", "norm.bias",
                                      "LayerNorm.weight", "LayerNorm.bias"]
            )
            group = {
                "params": [param],
                "lr": base_lr * (backbone_multiplier if is_backbone else 1.0),
                "weight_decay": weight_decay_norm if no_decay else weight_decay,
            }
            (backbone_params if is_backbone else other_params).append(group)

        param_groups = backbone_params + other_params
        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                param_groups, lr=base_lr, momentum=cfg.SOLVER.MOMENTUM
            )
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer_type}")

        # Gradient clipping (opcional, útil con transformers)
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    # ------------------------------------------------------------------
    # Test Time Augmentation (opcional, mejora ~1-2pp en métricas)
    # ------------------------------------------------------------------
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger.info("Evaluando con Test Time Augmentation (TTA)...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(
                cfg.OUTPUT_DIR, "inference_TTA"
            ))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


# ---------------------------------------------------------------------------
# Setup de configuración
# ---------------------------------------------------------------------------
def setup(args):
    cfg = get_cfg()

    # Añadir nodos de configuración de cada componente
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_metaformer_config(cfg)   # tus nodos MODEL.METAFORMER.*

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="mask2former_metaformer",
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    cfg = setup(args)

    # Modo evaluación pura (--eval-only)
    if args.eval_only:
        model = MetaformerTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MetaformerTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(MetaformerTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Entrenamiento normal
    trainer = MetaformerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    logger.info("Argumentos: " + str(args))

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
