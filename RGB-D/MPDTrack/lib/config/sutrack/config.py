from easydict import EasyDict as edict
import yaml

'''
SUTrack
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.USE_STARTRACK = False
cfg.MODEL.STARTRACK_CKPT = "checkpoints/startrack_mamba_diff_full/last.pth"
cfg.MODEL.STARTRACK_TOPK = 8
cfg.MODEL.STARTRACK_HISTORY_LEN = 32
cfg.MODEL.STARTRACK_UPDATE_RATIO_THRESH = 0.90
cfg.MODEL.STARTRACK_TARGET_PROB_THRESH = 0.50
cfg.MODEL.STARTRACK_MIN_ID_MARGIN = 0.00
cfg.MODEL.STARTRACK_VERBOSE = False
cfg.MODEL.STARTRACK_STATE_MODE = "shadow"
# cfg.MODEL.STARTRACK_DRIFT_GATE_ENABLE = False
# cfg.MODEL.STARTRACK_MAX_BASELINE_CENTER_JUMP = 0.75
# cfg.MODEL.STARTRACK_MAX_STATE_CENTER_JUMP = 1.50
# cfg.MODEL.STARTRACK_MAX_SIZE_RATIO = 2.50
# cfg.MODEL.STARTRACK_MIN_SIZE_RATIO = 0.40
# cfg.MODEL.STARTRACK_MAX_AREA_RATIO = 4.00
# cfg.MODEL.STARTRACK_MIN_AREA_RATIO = 0.25
# cfg.MODEL.STARTRACK_DRIFT_BYPASS_PROB = 0.90
# cfg.MODEL.STARTRACK_DRIFT_BYPASS_MARGIN = 0.40
# cfg.MODEL.STARTRACK_MIN_BASELINE_IOU_FOR_SWITCH = 0.05
# cfg.MODEL.STARTRACK_REJECT_COOLDOWN = 2
# STARTrack output-level correction policy
cfg.MODEL.STARTRACK_APPLY_PROB_THRESH = 0.65
cfg.MODEL.STARTRACK_APPLY_MARGIN_THRESH = 0.05
cfg.MODEL.STARTRACK_APPLY_AMBIGUITY_MIN = 0.30
cfg.MODEL.STARTRACK_APPLY_MIN_HISTORY = 6

# TAKS_INDEX
cfg.MODEL.TASK_NUM=5 #should be the largest index number + 1
cfg.MODEL.TASK_INDEX = edict() # index for tasks
cfg.MODEL.TASK_INDEX.VASTTRACK = 0
cfg.MODEL.TASK_INDEX.LASOT = 0
cfg.MODEL.TASK_INDEX.TRACKINGNET = 0
cfg.MODEL.TASK_INDEX.GOT10K = 0
cfg.MODEL.TASK_INDEX.COCO = 0
cfg.MODEL.TASK_INDEX.TNL2K = 1
cfg.MODEL.TASK_INDEX.DEPTHTRACK = 2
cfg.MODEL.TASK_INDEX.LASHER = 3
cfg.MODEL.TASK_INDEX.VISEVENT = 4


# MODEL.LANGUAGE
cfg.MODEL.TEXT_ENCODER = edict()
cfg.MODEL.TEXT_ENCODER.TYPE = 'ViT-L/14' # clip: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

# MODEL.ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "fastitpnb" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "pretrained/itpn/fast_itpn_base_clipl_e1600.pt" #
cfg.MODEL.ENCODER.PATCHEMBED_INIT = "halfcopy" # copy, halfcopy, random
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
cfg.MODEL.ENCODER.STRIDE = 14
cfg.MODEL.ENCODER.POS_TYPE = 'index' # type of loading the positional encoding. "interpolate" or "index".
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = True # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.CLASS_TOKEN = True # class token

# MODEL.ENCODER_POSTPROCESS
cfg.MODEL.ENCODER_POSTPROCESS = edict()
cfg.MODEL.ENCODER_POSTPROCESS.ENABLED = False
cfg.MODEL.ENCODER_POSTPROCESS.APPLY_ON_SEARCH_ONLY = True

cfg.MODEL.ENCODER_POSTPROCESS.TEMPORAL_MAMBA = edict()
cfg.MODEL.ENCODER_POSTPROCESS.TEMPORAL_MAMBA.ENABLED = False
cfg.MODEL.ENCODER_POSTPROCESS.TEMPORAL_MAMBA.D_STATE = 16
cfg.MODEL.ENCODER_POSTPROCESS.TEMPORAL_MAMBA.EXPAND = 2
cfg.MODEL.ENCODER_POSTPROCESS.TEMPORAL_MAMBA.USE_ONLINE_MEMORY_INFERENCE = False

# MODEL.POST_DECODER_DISAMBIGUATOR
cfg.MODEL.POST_DECODER_DISAMBIGUATOR = edict()
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.ENABLED = False
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.RATIO_THRESH = 0.8
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.TOPK_PEAKS = 8
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.NMS_KERNEL_SIZE = 5
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.MULTI_PEAK_RATIO_THRESH = 0.45
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.GAUSSIAN_SIGMA = 2.0
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.SUPPRESSION_STRENGTH = 0.6
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.HISTORY_LEN = 32
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.USE_MAMBA_HISTORY = True
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.USE_MAMBA_HISTORY_BANK = True
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.MAMBA_D_STATE = 16
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.MAMBA_EXPAND = 2
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.TEMPLATE_FEAT_DIM = 512
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.USE_TEMPLATE_ANCHOR = False
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.USE_FIRST_FRAME_ANCHOR = False
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.TEMPLATE_ANCHOR_WEIGHT = 0.35
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.FIRST_FRAME_ANCHOR_WEIGHT = 0.40
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.MAMBA_HISTORY_WEIGHT = 0.25
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.USE_HISTORY_AWARE_RERANK_SCORE = False
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.HISTORY_RERANK_WEIGHT = 1.0
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.TARGET_LOGIT_WEIGHT = 1.0
cfg.MODEL.POST_DECODER_DISAMBIGUATOR.PEAK_SCORE_WEIGHT = 0.2

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256
cfg.MODEL.DECODER.CONV_TYPE = "normal" # normal: 3*3 conv, small: 1*1 conv, only for the center head for now.
cfg.MODEL.DECODER.XAVIER_INIT = True

# MODEL.TASK_DECODER
cfg.MODEL.TASK_DECODER = edict()
cfg.MODEL.TASK_DECODER.NUM_CHANNELS = 256
cfg.MODEL.TASK_DECODER.FEATURE_TYPE = "average" # class: using class token, average: average the feature, text: using the text token

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 180
cfg.TRAIN.LR_DROP_EPOCH = 144
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.ENCODER_POSTPROCESS_MULTIPLIER = 1.0
cfg.TRAIN.FREEZE_ENCODER = False # for freezing the parameters of encoder
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.TASK_CE_WEIGHT = 1.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.COMPACT_LOG = True
cfg.TRAIN.FULL_LOG_INTERVAL = 5
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
cfg.TRAIN.SAVE_EVERY_EPOCH = False
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft, fft, text_frozen
cfg.TRAIN.PRETRAINED_PATH = None
cfg.TRAIN.TWO_STAGE = edict()
cfg.TRAIN.TWO_STAGE.ENABLED = False
cfg.TRAIN.TWO_STAGE.STAGE1_EPOCHS = 5
cfg.TRAIN.TWO_STAGE.FREEZE_BACKBONE_IN_STAGE1 = True
cfg.TRAIN.TWO_STAGE.FREEZE_DECODER_IN_STAGE1 = True
cfg.TRAIN.TWO_STAGE.FREEZE_TASK_DECODER_IN_STAGE1 = True
cfg.TRAIN.TWO_STAGE.FREEZE_TEXT_ENCODER_IN_STAGE1 = True
cfg.TRAIN.TWO_STAGE.KEEP_BACKBONE_FROZEN_IN_STAGE2 = True
cfg.TRAIN.TWO_STAGE.UNFREEZE_DECODER_IN_STAGE2 = True
cfg.TRAIN.TWO_STAGE.UNFREEZE_TASK_DECODER_IN_STAGE2 = True
cfg.TRAIN.THREE_STAGE = edict()
cfg.TRAIN.THREE_STAGE.ENABLED = False
cfg.TRAIN.THREE_STAGE.STAGE1_EPOCHS = 5
cfg.TRAIN.THREE_STAGE.STAGE2_EPOCHS = 5
cfg.TRAIN.THREE_STAGE.DECODER_LAST_N_BLOCKS = 2
cfg.TRAIN.THREE_STAGE.LAMBDA_GATE = 0.1
cfg.TRAIN.THREE_STAGE.USE_TRACK_LOSS_STAGE2 = False
cfg.TRAIN.THREE_STAGE.STAGE3_FREEZE_SUTRACK = True

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.MULTI_MODAL_VISION = True # vision multi-modal
cfg.DATA.MULTI_MODAL_LANGUAGE = True # language multi-modal
cfg.DATA.USE_NLP = edict() # using the text of the dataset
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.REFCOCOG = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.OTB99 = False
cfg.DATA.USE_NLP.DEPTHTRACK = False
cfg.DATA.USE_NLP.LASHER = False
cfg.DATA.USE_NLP.VISEVENT = False
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  #number of search region, only support 1 for now.
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 2.0
cfg.TEST.SEARCH_SIZE = 128
cfg.TEST.EPOCH = 180
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 999999
#
cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 1.0
#
cfg.TEST.MULTI_MODAL_VISION = edict()
cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True
#
cfg.TEST.MULTI_MODAL_LANGUAGE = edict()
cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False
#
cfg.TEST.USE_NLP = edict()
cfg.TEST.USE_NLP.DEFAULT = False
cfg.TEST.USE_NLP.TNL2K = True





def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k not in base_cfg and isinstance(k, str) and k.upper() in base_cfg:
                k = k.upper()
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    import yaml
    from easydict import EasyDict as edict

    filename = str(filename)

    with open(filename, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if raw_cfg is None:
        raise ValueError(f"Empty YAML config file: {filename}")

    if not isinstance(raw_cfg, dict):
        raise TypeError(
            f"YAML config must be a dict, but got {type(raw_cfg)} from {filename}"
        )

    exp_config = edict(raw_cfg)

    _update_config(cfg, exp_config)

