import os
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, Imagenet1k, VastTrack
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent, LasHeR, DepthTrack
from lib.train.dataset import Otb99_lang, Tnl2k, RefCOCOSeq
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def _read_subset_spec(spec_file):
    if not os.path.exists(spec_file):
        raise FileNotFoundError(f"Subset spec file not found: {spec_file}")
    entries = []
    with open(spec_file, "r") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue
            entries.append(line)
    if len(entries) == 0:
        raise ValueError(f"Subset spec file is empty: {spec_file}")
    return entries


def _filter_dataset_by_subset(dataset, subset_entries, subset_tag):
    if not hasattr(dataset, "sequence_list"):
        raise ValueError(f"Dataset {type(dataset).__name__} does not support subset filtering.")

    sequence_list = list(dataset.sequence_list)
    if len(sequence_list) == 0:
        raise ValueError(f"Dataset {type(dataset).__name__} has empty sequence_list before filtering.")

    keep_indices = []
    for entry in subset_entries:
        if entry.isdigit():
            index = int(entry)
            if 0 <= index < len(sequence_list):
                keep_indices.append(index)
            continue
        try:
            keep_indices.append(sequence_list.index(entry))
        except ValueError:
            continue

    keep_indices = sorted(set(keep_indices))
    if len(keep_indices) == 0:
        raise ValueError(f"Subset {subset_tag} selected zero sequences in {type(dataset).__name__}.")

    dataset.sequence_list = [sequence_list[index] for index in keep_indices]

    if hasattr(dataset, "sequence_meta_info"):
        keep_names = set(dataset.sequence_list)
        dataset.sequence_meta_info = {
            name: value for name, value in dataset.sequence_meta_info.items() if name in keep_names
        }

    if hasattr(dataset, "_build_seq_per_class"):
        dataset.seq_per_class = dataset._build_seq_per_class()
        if hasattr(dataset, "class_list"):
            dataset.class_list = list(dataset.seq_per_class.keys())
            dataset.class_list.sort()

    if hasattr(dataset, "_build_class_list"):
        dataset.seq_per_class = dataset._build_class_list()

    print(f"[Subset] {subset_tag}: keep {len(dataset.sequence_list)} sequences.")


def _resolve_dataset_name(name):
    specs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_specs')
    subset_map = {
        "LASOT_HARD": ("LASOT", os.path.join(specs_dir, "lasot_hard.txt")),
        "GOT10K_vottrain_HARD": ("GOT10K_vottrain", os.path.join(specs_dir, "got10k_vottrain_hard.txt")),
        "TNL2K_train_HARD": ("TNL2K_train", os.path.join(specs_dir, "tnl2k_train_hard.txt")),
    }
    if name in subset_map:
        return subset_map[name]
    return name, None

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.compact_log = bool(getattr(cfg.TRAIN, "COMPACT_LOG", True))
    settings.full_log_interval = int(getattr(cfg.TRAIN, "FULL_LOG_INTERVAL", 5))
    settings.save_every_epoch = getattr(cfg.TRAIN, "SAVE_EVERY_EPOCH", False)
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search':getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.multi_modal_vision = getattr(cfg.DATA, "MULTI_MODAL_VISION", False)
    settings.multi_modal_language = getattr(cfg.DATA, "MULTI_MODAL_LANGUAGE", False)
    settings.use_nlp = cfg.DATA.USE_NLP
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        settings.fix_norm = True
    else:
        settings.fix_norm = False


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full",
                        "COCO17", "VID", "TRACKINGNET", "IMAGENET1K",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train","LasHeR_val", "VisEvent",
                        "REFCOCOG", "TNL2K_train", "OTB99_train","VASTTRACK",
                        "LASOT_HARD", "GOT10K_vottrain_HARD", "TNL2K_train_HARD"]
        resolved_name, subset_file = _resolve_dataset_name(name)
        if resolved_name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader,
                                           multi_modal_vision=settings.multi_modal_vision,
                                           multi_modal_language=settings.multi_modal_language,
                                           use_nlp=settings.use_nlp['LASOT']))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language,
                                      use_nlp=settings.use_nlp['LASOT']))
        if resolved_name == "VASTTRACK":
            datasets.append(VastTrack(settings.env.vasttrack_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language,
                                      use_nlp=settings.use_nlp['VASTTRACK']))
        if resolved_name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if resolved_name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if resolved_name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if resolved_name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader,
                                               multi_modal_vision=settings.multi_modal_vision,
                                               multi_modal_language=settings.multi_modal_language,
                                               use_nlp=settings.use_nlp['COCO']
                                               ))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader,
                                          multi_modal_vision=settings.multi_modal_vision,
                                          multi_modal_language=settings.multi_modal_language,
                                          use_nlp=settings.use_nlp['COCO']
                                          ))
        if resolved_name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if resolved_name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader,
                                                 multi_modal_vision=settings.multi_modal_vision,
                                                 multi_modal_language=settings.multi_modal_language,
                                                 use_nlp=settings.use_nlp['TRACKINGNET']
                                                 ))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['TRACKINGNET']
                                            ))
        if resolved_name == "IMAGENET1K":
            datasets.append(Imagenet1k(settings.env.imagenet1k_dir, image_loader=image_loader))
        if resolved_name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir,
                                       dtype='color' if not settings.multi_modal_vision else 'rgbcolormap',
                                       split='train',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['DEPTHTRACK']
                                       ))
        if resolved_name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir,
                                       dtype='color' if not settings.multi_modal_vision else 'rgbcolormap',
                                       split='val',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['DEPTHTRACK']
                                       ))
        if resolved_name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='all',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if resolved_name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='train',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if resolved_name == "LasHeR_val":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='val',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if resolved_name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir,
                                     dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                     split='train',
                                     multi_modal_vision=settings.multi_modal_vision,
                                     multi_modal_language=settings.multi_modal_language,
                                     use_nlp=settings.use_nlp['VISEVENT']
                                     ))
        if resolved_name == "REFCOCOG":
            datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split="train", image_loader=image_loader,
                                       name="refcocog", splitBy="google",
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['REFCOCOG']
                                       ))
        if resolved_name == "TNL2K_train":
            datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader,
                                  multi_modal_vision=settings.multi_modal_vision,
                                  multi_modal_language=settings.multi_modal_language,
                                  use_nlp=settings.use_nlp['TNL2K']
                                  ))
        elif resolved_name == "OTB99_train":
            datasets.append(Otb99_lang(settings.env.otb99_dir, split='train', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['OTB99']
                                       ))

        if subset_file is not None:
            subset_entries = _read_subset_spec(subset_file)
            _filter_dataset_by_subset(datasets[-1], subset_entries, name)

    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          multi_modal_language=settings.multi_modal_language,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    # print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode,
                                            multi_modal_language=settings.multi_modal_language
                                            )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def build_optimizer(model, weight_decay=1e-4):
    """
    Build stage-friendly layer-wise AdamW optimizer.

    Rules:
      - encoder:            lr = 1e-6
      - decoder:            lr = 5e-6
      - head:               lr = 1e-5
      - post_disambiguator: lr = 1e-4
    """
    # 1) DDP/DataParallel compatibility.
    net = model.module if hasattr(model, "module") else model

    module_specs = [
        ("encoder", "encoder", 1e-6),
        ("decoder", "decoder", 5e-6),
        ("head", "head", 1e-5),
        ("post_disambiguator", "post_disambiguator", 1e-4),
    ]
    # Compatibility fallback for current codebase naming.
    if (not hasattr(net, "head")) and hasattr(net, "task_decoder"):
        module_specs[2] = ("head(task_decoder)", "task_decoder", 1e-5)

    param_groups = []
    seen_params = set()

    for group_name, attr_name, group_lr in module_specs:
        module = getattr(net, attr_name, None)
        if module is None:
            continue

        group_params = []
        num_params = 0
        for parameter in module.parameters():
            # 3) only trainable params.
            if not parameter.requires_grad:
                continue
            # 5) avoid duplicates.
            pid = id(parameter)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            group_params.append(parameter)
            num_params += int(parameter.numel())

        # 4) skip empty group.
        if len(group_params) == 0:
            continue

        param_groups.append({
            "name": group_name,
            "params": group_params,
            "lr": group_lr,
        })

        # 7) print group summary.
        if is_main_process():
            print(f"[Optimizer] group={group_name}, lr={group_lr:.1e}, params={num_params}")

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found. Please check stage freeze settings.")

    # 6) AdamW
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def get_optimizer_scheduler(net, cfg):
    def append_param_group(groups, params, lr=None):
        params = [param for param in params if param.requires_grad]
        if len(params) == 0:
            return
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        groups.append(group)

    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]},
        ]
        for n, p in net.named_parameters():
            if "prompt" not in n:
                p.requires_grad = False

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "fft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR / cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "text_frozen":
        for n, p in net.named_parameters():
            if ("clip" in n) or ("bert" in n):
                p.requires_grad = False

        param_dicts = []
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters()
             if ("encoder" not in n) and ("encoder_postprocess" not in n)],
        )
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters() if "encoder_postprocess" in n],
            lr=cfg.TRAIN.LR * getattr(cfg.TRAIN, "ENCODER_POSTPROCESS_MULTIPLIER", 1.0),
        )
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters()
             if ("encoder" in n) and ("encoder_postprocess" not in n) and ("clip" not in n)],
            lr=cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
        )

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    else:
        param_dicts = []
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters()
             if ("encoder" not in n) and ("encoder_postprocess" not in n)],
        )
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters() if "encoder_postprocess" in n],
            lr=cfg.TRAIN.LR * getattr(cfg.TRAIN, "ENCODER_POSTPROCESS_MULTIPLIER", 1.0),
        )
        append_param_group(
            param_dicts,
            [p for n, p in net.named_parameters()
             if ("encoder" in n) and ("encoder_postprocess" not in n)],
            lr=cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
        )
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        use_three_stage = bool(getattr(getattr(cfg.TRAIN, "THREE_STAGE", None), "ENABLED", False))
        if use_three_stage:
            optimizer = build_optimizer(net, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
