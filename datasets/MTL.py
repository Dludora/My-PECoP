# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/2 16:19
import copy
import glob
import os
import pickle
import numpy as np
import random
import cv2
from utils.util import read_pkl
from torchvision import transforms
import torch
from torch.utils.data import Dataset


class MTL_Dataset(Dataset):
    """MTL-AQA dataset"""

    def __init__(self, args, transforms_=None, color_jitter_=None, subset="train"):
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

        self.rgb_prefix = args.rgb_prefix
        self.clip_len = args.clip_len
        self.max_sample_rate = args.max_sample_rate
        self.max_segment = args.max_segment
        self.fr = args.fr
        self.toPIL = transforms.ToPILImage()
        self.transforms_ = transforms_
        self.color_jitter_ = color_jitter_

    def get_start_last_frame(self, video_name):
        end_frame = self.label_dict.get(video_name).get("end_frame")
        start_frame = self.label_dict.get(video_name).get("start_frame")
        frame_num = end_frame - start_frame + 1

        return start_frame, end_frame, frame_num

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

        rgb_clip = self.load_clip(
            sample,
            sample_start_frame,
            sample_rate=sample_rate,
            clip_len=self.clip_len,
            num_frames=sample_frame_num,
            segment_start_frame=segment_start_frame,
            segment_last_frame=segment_last_frame,
        )

        label_speed = sample_rate - 1
        label_segment = segment - 1
        label = [label_speed, label_segment]

        trans_clip = self.transforms_(rgb_clip)

        return trans_clip, np.array(label)

    def load_clip(
        self,
        video_file_name,
        start_frame,
        sample_rate,
        clip_len,
        num_frames,
        segment_start_frame,
        segment_last_frame,
    ):
        video_clip = []
        idx1 = 0
        idx = 0
        clip_start_frame = copy.deepcopy(start_frame)
        normal_f = copy.deepcopy(start_frame)

        for i in range(clip_len):
            if segment_start_frame <= i <= segment_last_frame:
                cur_img_path = os.path.join(
                    self.data_root,
                    "new",
                    "{:02}/{:08}.jpg".format(
                        video_file_name[0], start_frame + idx1 * sample_rate
                    ),
                )
                normal_f = start_frame + (idx1 * sample_rate)
                idx = 1

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (
                    start_frame + (idx1 + 1) * sample_rate
                ) > clip_start_frame + num_frames - 1:
                    start_frame = 1
                    normal_f = 1
                    idx = 0
                    idx1 = 0
                else:
                    idx1 += 1
            else:
                cur_img_path = os.path.join(
                    self.data_root,
                    "new",
                    "{:02}/{:08}.jpg".format(video_file_name[0], normal_f + idx),
                )

                start_frame = normal_f + idx
                idx1 = 1

                # print(cur_img_path)

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (normal_f + (idx + 4)) > clip_start_frame + num_frames - 1:
                    normal_f = 1
                    start_frame = 1
                    idx1 = 0
                    idx = 0
                else:
                    idx += self.fr

        video_clip = np.array(video_clip)

        return video_clip

    def __len__(self):
        return len(self.dataset)
