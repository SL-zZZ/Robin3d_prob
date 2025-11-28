import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence
import re
import random
from collections import OrderedDict
from transformers import GPT2Tokenizer
logger = logging.getLogger(__name__)

class TrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, dataset_name, config, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num

        mask3d_feat_pos_file, mask3d_feat_file, feat_file, img_feat_file, attribute_file, anno_file, gt_attribute_file, dca_file = ann_list[:8]
        self.attributes = torch.load(attribute_file, map_location='cpu')
        self.gt_attributes = torch.load(gt_attribute_file, map_location='cpu')
        self.anno = json.load(open(anno_file, 'r'))
        
        self.dca_anno = json.load(open(dca_file, "r"))
        self.gt2_tokenizer = GPT2Tokenizer.from_pretrained("/home/dell/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
        self.gt2_tokenizer.pad_token = self.gt2_tokenizer.eos_token
        
        if mask3d_feat_pos_file == "":
            mask3d_feat_pos_file = None
        if mask3d_feat_file == "":
            mask3d_feat_file = None


        
        if feat_file in TrainDataset.cached_feats and img_feat_file in TrainDataset.cached_feats:
            self.scene_feats = TrainDataset.cached_feats[feat_file]
            self.scene_img_feats = TrainDataset.cached_feats[img_feat_file]
            self.scene_mask3d_feats = TrainDataset.cached_feats[mask3d_feat_file]
            self.scene_mask3d_feat_poses = TrainDataset.cached_feats[mask3d_feat_pos_file]
            self.scene_valid_mask3d = TrainDataset.cached_feats[attribute_file]
        else:
            if mask3d_feat_pos_file is not None and os.path.exists(mask3d_feat_pos_file):
                self.mask3d_feat_pos = torch.load(mask3d_feat_pos_file, map_location='cpu')
            else:
                self.mask3d_feat_pos = None
            if mask3d_feat_file is not None and os.path.exists(mask3d_feat_file):
                self.mask3d_feats = torch.load(mask3d_feat_file, map_location='cpu')
            else:
                self.mask3d_feats = None
            if feat_file is not None and os.path.exists(feat_file):
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = None
                self.scene_valid_mask3d = None
            else:
                self.scene_mask3d_feat_poses, self.scene_mask3d_feats, self.scene_feats, self.scene_img_feats, self.scene_valid_mask3d = self.prepare_scene_features_dca()
            TrainDataset.cached_feats[feat_file] = self.scene_feats
            TrainDataset.cached_feats[img_feat_file] = self.scene_img_feats
            TrainDataset.cached_feats[mask3d_feat_file] = self.scene_mask3d_feats
            TrainDataset.cached_feats[mask3d_feat_pos_file] = self.scene_mask3d_feat_poses
            TrainDataset.cached_feats[attribute_file] = self.scene_valid_mask3d



    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            return self.__getitem__(random.randint(0, len(self.anno)-1))
        obj_id = self.anno[index].get('obj_id', 0)
        scene_id = self.anno[index].get("scene_id", None)
        scene_caption = []
        if scene_id is not None:
            for obj_id in range(self.max_obj_num):
                scene_obj = f"{scene_id}_{obj_id}"
                if scene_obj in self.dca_anno:
                    scene_caption.append(random.choice(self.dca_anno[scene_obj])+self.gt2_tokenizer.eos_token)
                else:
                    scene_caption.append('')
        else:
            scene_caption = [''] * self.max_obj_num
        dca_dict = self.get_token_ids_dca(scene_caption)

        
        if isinstance(obj_id, str):
            obj_id = int(obj_id)
        if isinstance(obj_id, list):
            obj_id = [int(i) for i in obj_id]
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        caption = self.anno[index]["caption"]

        scene_mask3d_feat_pos, scene_mask3d_feat, scene_feat, scene_img_feat, scene_id = self.get_anno(index)

        click = torch.from_numpy(np.array(self.anno[index].get("click", torch.zeros(3))))
        point_min = torch.from_numpy(np.array(self.anno[index].get("min", torch.zeros(3))))
        point_max = torch.from_numpy(np.array(self.anno[index].get("max", torch.ones(3))))
        
        # get token ids and mask
        res_dict = self.get_token_ids_mask(question, caption, obj_id)

        # get special str index
        click_token_index = self.get_special_str_index('<click>', res_dict['all_token_id'])


        return res_dict["all_token_id"], res_dict["all_token_mask"], res_dict["tgt_idx"], res_dict["obj_mask"], \
                scene_mask3d_feat_pos, scene_mask3d_feat, scene_feat, scene_img_feat, \
                    click, point_min, point_max, click_token_index, dca_dict["dca_token_id"], dca_dict["dca_token_mask"] #, res_dict["scale_factor"]


def train_collate_fn(batch):
    all_token_id, all_token_mask, tgt_idx, obj_mask, \
        scene_mask3d_feat_pos, scene_mask3d_feat, scene_feat, scene_img_feat, \
        click, point_min, point_max, click_token_index, dca_token_id, dca_token_mask = zip(*batch)
    
    return {
        "all_token_id": pad_sequence(all_token_id, batch_first=True, padding_value=0),
        "all_token_mask": pad_sequence(all_token_mask, batch_first=True, padding_value=0),
        "tgt_idx": pad_sequence(tgt_idx, batch_first=True, padding_value=-100),
        "obj_mask": pad_sequence(obj_mask, batch_first=True, padding_value=0),

        "scene_mask3d_feat_pos": pad_sequence(scene_mask3d_feat_pos, batch_first=True),
        "scene_mask3d_feat": pad_sequence(scene_mask3d_feat, batch_first=True),
        "scene_feat": pad_sequence(scene_feat, batch_first=True),
        "scene_img_feat": pad_sequence(scene_img_feat, batch_first=True),
        "clicks": pad_sequence(click, batch_first=True),
        "point_mins": pad_sequence(point_min, batch_first=True),
        "point_maxs": pad_sequence(point_max, batch_first=True),
        "click_token_index": pad_sequence(click_token_index, batch_first=True, padding_value=0),

        # "scale_factor": torch.stack(scale_factor),

        "training": True,
        "dca_token_id": pad_sequence(dca_token_id, batch_first=True, padding_value=0),
        "dca_token_mask": pad_sequence(dca_token_mask, batch_first=True, padding_value=0),
    }
