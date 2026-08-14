"""
GroundingDINO 训练配置 - v5 P0优化版
 - 目标: 学习特征→位置映射，不记忆固定位置
 - 关键改动 (P0优化):
   1. BERT 加载预训练权重 (通过 hf-mirror.com)
   2. matcher/loss 平衡: cost_class=5, cost_bbox=8, cost_giou=6 (在 groundingdino.py 中设)
   3. NUM_QUERIES 10→50, BASE_LR 1.5e-5→8e-6, warmup 保持 8 epochs
   4. box_head bias 初始化为小目标尺寸 (w≈0.1, h≈0.2)
   5. 几何数据增强: 翻转+缩放(0.7~1.3)+平移(±15%) 打破位置偏置
"""
from groundingdino.config.defaults import _C as cfg

cfg.MODEL.NAME = "groundingdino_multi_modal"

cfg.MODEL.BACKBONE.NAME = "swin_T_224_1k"
cfg.MODEL.BACKBONE.OUT_CHANNELS = [96, 192, 384, 768]
cfg.MODEL.USE_SWIN = True
cfg.MODEL.PRETRAINED = False

cfg.MODEL.NECK.NAME = "MultiScaleDeformableAttention"
cfg.MODEL.NECK.IN_CHANNELS = [96, 192, 384, 768]
cfg.MODEL.NECK.OUT_CHANNEL = 256
cfg.MODEL.NECK.NUM_LAYERS = 2
cfg.MODEL.NECK.NUM_HEADS = 4

cfg.MODEL.DECODER.NAME = "GroundingDINOTransformerDecoder"
cfg.MODEL.DECODER.NUM_LAYERS = 2
cfg.MODEL.DECODER.NUM_HEADS = 4
cfg.MODEL.DECODER.DIM_FEEDFORWARD = 512

cfg.MODEL.TEXT_ENCODER.NAME = "bert-base-uncased"
cfg.MODEL.TEXT_ENCODER.DIM = 768
cfg.MODEL.MAX_SEQ_LEN = 256

cfg.MODEL.MULTI_MODAL.ENABLED = True
cfg.MODEL.MULTI_MODAL.MODALITIES = ["rgb", "ir", "depth"]
cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_RGB = 3
cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_IR = 1
cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_DEPTH = 1
cfg.MODEL.MULTI_MODAL.ADAPTER_DIM = 96
cfg.MODEL.MULTI_MODAL.USE_GATED_FUSION = True
cfg.MODEL.MULTI_MODAL.USE_TEXT_GUIDANCE = True

cfg.MODEL.MASK_ON = False
cfg.MODEL.NUM_QUERIES = 50
cfg.MODEL.HIDDEN_DIM = 64

# P0-3: 更低的学习率，配合 BERT 预训练权重稳定微调
cfg.SOLVER.BASE_LR = 8e-6
cfg.SOLVER.WEIGHT_DECAY = 1e-3
cfg.SOLVER.EPOCHS = 100
cfg.SOLVER.BATCH_SIZE = 8
cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
cfg.SOLVER.WARMUP_EPOCHS = 8
cfg.SOLVER.AMP = True
cfg.SOLVER.NUM_WORKERS = 0

cfg.DATASETS.TRAIN = ["multi_modal_train"]
cfg.DATASETS.TEST = ["multi_modal_val"]
cfg.DATASETS.DATA_PATH = "./data"
cfg.DATASETS.TRAIN_FILE = "train.json"
cfg.DATASETS.VAL_FILE = "val.json"
cfg.DATASETS.MAX_TRAIN_SAMPLES = -1
cfg.DATASETS.MAX_VAL_SAMPLES = -1

cfg.INPUT.SIZE_TRAIN = (512, 512)
cfg.INPUT.SIZE_TEST = (512, 512)

cfg.OUTPUT_DIR = "./output"
cfg.LOG_DIR = "./logs"

cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.SAVE_BEST_ONLY = True
cfg.TEST.EARLY_STOPPING_PATIENCE = 25

cfg.SEED = 42


def get_cfg():
    return cfg.clone()
