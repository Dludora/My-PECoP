# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/2 16:19
import copy
import glob
import os
import pickle
import numpy as np
import random
from PIL import Image
from utils.util import read_pkl
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import ipdb


class MTL_Dataset(Dataset):
    """MTL-AQA dataset"""

    def __init__(self, args, transforms=None, color_jitter=None, subset="train"):
        super(MTL_Dataset, self).__init__()
        self.data_root = os.path.join(args.data_root)
        # get_data
        if subset == "test":
            self.dataset = read_pkl(
                os.path.join(self.data_root, "info/test_split_0.pkl")
            )
        else:
            self.dataset = read_pkl(
                os.path.join(self.data_root, "info/train_split_0.pkl")
            )
        # get label
        self.label_dict = read_pkl(
            os.path.join(
                self.data_root, "info/final_annotations_dict_with_dive_number.pkl"
            )
        )
        self.subset = subset

        self.clip_len = args.clip_len
        self.max_sample_rate = args.max_sample_rate
        self.max_segment = args.max_segment
        self.fr = args.fr
        self.transforms_ = transforms
        self.color_jitter_ = color_jitter

    def get_start_last_frame(self, video_name):
        end_frame = self.label_dict.get(video_name).get("end_frame")
        start_frame = self.label_dict.get(video_name).get("start_frame")
        frame_num = end_frame - start_frame + 1

        return start_frame, end_frame, frame_num

    def load_video(
        self,
        video,
        start_frame,
        end_frame,
        sample_rate,
        frame_len,
        segment_start_frame,
        segment_last_frame,
    ):
        video_path = os.path.join(self.data_root, "new")

        video_clip = []
        idx = start_frame
        cur_len = 0

        while cur_len < self.clip_len:
            img_path = os.path.join(video_path, "{:02}/{:08}.jpg".format(video[0], idx))
            cur_len += 1
            idx += (
                sample_rate
                if segment_start_frame <= cur_len <= segment_last_frame
                else self.fr
            )
            if idx > end_frame:
                idx = start_frame + (idx - end_frame - 1)
            img = Image.open(img_path)
            video_clip.append(img)

        if self.subset == "train":
            video_clip = [self.color_jitter_(img) for img in video_clip]

        video_clip = self.transforms_(video_clip)

        return video_clip

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_start_frame, sample_end_frame, sample_frame_num = (
            self.get_start_last_frame(sample)
        )
        # playback speed of the random clip
        sample_rate = random.randint(1, self.max_sample_rate)
        while sample_rate == self.fr:
            sample_rate = random.randint(1, self.max_sample_rate)

        # segment of the random clip
        segment = random.randint(1, self.max_segment)

        # 选中的segment的起始帧和结束帧
        segment_start_frame = int((segment - 1) * (self.clip_len / self.max_segment))
        segment_last_frame = int(segment * (self.clip_len / self.max_segment))

        video_clip = self.load_video(
            sample,
            sample_start_frame,
            sample_end_frame,
            sample_rate=sample_rate,
            frame_len=sample_frame_num,
            segment_start_frame=segment_start_frame,
            segment_last_frame=segment_last_frame,
        )

        label_speed = sample_rate - 1
        label_segment = segment - 1
        label = [label_speed, label_segment]

        return video_clip, np.array(label)

    def __len__(self):
        return len(self.dataset)
