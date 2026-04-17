import os
import os.path
import json
import torch
import numpy as np
import pandas
import csv
import random
import cv2
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from lib.train.data.image_loader import hdf5_depth_loader


class Lasot(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False, use_nlp=False, sequence_filter_file=None,
                 split_file=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dir if root is None else root
        super().__init__('LaSOT', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split, split_file)
        if sequence_filter_file:
            self.sequence_list = self._apply_sequence_filter(self.sequence_list, sequence_filter_file)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp

    def _load_sequence_filter(self, sequence_filter_file):
        file_ext = os.path.splitext(sequence_filter_file)[1].lower()
        if file_ext == '.json':
            with open(sequence_filter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return sorted({item['sequence'] for item in data if 'sequence' in item})
        if file_ext == '.txt':
            with open(sequence_filter_file, 'r', encoding='utf-8') as f:
                return sorted({line.strip() for line in f if line.strip()})
        raise ValueError(f"Unsupported sequence filter file: {sequence_filter_file}")

    def _apply_sequence_filter(self, sequence_list, sequence_filter_file):
        filtered_names = set(self._load_sequence_filter(sequence_filter_file))
        filtered_list = [seq for seq in sequence_list if seq in filtered_names]
        print(f"[LaSOT] Sequence filter enabled: {len(filtered_list)}/{len(sequence_list)} sequences kept from {sequence_filter_file}")
        return filtered_list

    def _resolve_split_file(self, split, split_file=None):
        if split_file:
            return split_file

        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        if split == 'train':
            return os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
        if split and os.path.isfile(split):
            return split
        if split:
            candidate_paths = [
                os.path.join(ltr_path, 'data_specs', f'{split}.txt'),
                os.path.join(ltr_path, 'data_specs', f'lasot_{split}.txt'),
                os.path.join(ltr_path, 'data_specs', f'lasot_{split}_split.txt'),
            ]
            for candidate in candidate_paths:
                if os.path.isfile(candidate):
                    return candidate
        raise ValueError('Unknown split name.')

    def _build_sequence_list(self, vid_ids=None, split=None, split_file=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            file_path = self._resolve_split_file(split, split_file)
            # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language and self.use_nlp:
            nlp = self._read_nlp(seq_path)
            output['nlp'] = nlp
        return output

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    # def _get_frame(self, seq_path, frame_id):
    #     frame = self.image_loader(self._get_frame_path(seq_path, frame_id))
    #     if self.multi_modal_vision:
    #         frame = np.concatenate((frame, frame), axis=-1)
    #     return frame
    def _get_frame(self, seq_path, frame_id):
        img_path = self._get_frame_path(seq_path, frame_id)
        frame = self.image_loader(img_path)
        
        if self.multi_modal_vision:
            # 👑 1. 修改这里：匹配带后缀的新文件名
            depth_path = img_path.replace('.jpg', '_depth_l640.h5')
            
            depth = hdf5_depth_loader(depth_path)
            
            if depth is None:
                depth = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.float32)
            else:
                if depth.shape[0] != frame.shape[0] or depth.shape[1] != frame.shape[1]:
                    orig_W, orig_H = frame.shape[1], frame.shape[0]
                    # 👑 2. 放大时必须用 INTER_LINEAR，保持平滑过渡
                    depth = cv2.resize(depth, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
                    
                    if len(depth.shape) == 2:
                        depth = np.expand_dims(depth, axis=-1)
                
            frame = np.concatenate((frame, depth), axis=-1)
            
        return frame


    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    ###############################################################
    def _read_nlp(self, seq_path):
        nlp_file = os.path.join(seq_path, "nlp.txt")
        nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
        return nlp[0][0]

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames