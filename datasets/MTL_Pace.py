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


class MTL_Pace_Dataset(Dataset):

    def __init__(self, args, transforms=None, color_jitter=None, subset="train"):
        super(MTL_Pace_Dataset, self).__init__()
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
        self.transforms_ = transforms
        self.color_jitter_ = color_jitter

    def get_start_last_frame(self, video_name):
        end_frame = self.label_dict.get(video_name).get("end_frame")
        start_frame = self.label_dict.get(video_name).get("start_frame")
        frame_num = end_frame - start_frame + 1

        return start_frame, end_frame, frame_num

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len, num_frames):

        video_path = os.path.join(self.data_root, "new")
        video_clip = []
        idx = 0

        for i in range(clip_len):
            cur_img_path = os.path.join(
                video_path,
                "{:02}/{:08}.jpg".format(video_dir[0], start_frame + idx * sample_rate),
            )

            img = Image.open(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                idx = 0
            else:
                idx += 1

        if self.subset == "train":
            video_clip = [self.color_jitter_(img) for img in video_clip]

        video_clip = self.transforms_(video_clip)

        return video_clip

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_start_frame, sample_end_frame, sample_frame_num = (
            self.get_start_last_frame(sample)
        )
        sample_rate = random.randint(1, self.max_sample_rate)
        start_frame = random.randint(1, sample_frame_num)

        label = sample_rate - 1

        video_clip = self.loop_load_rgb(
            sample, sample_start_frame, sample_rate, self.clip_len, sample_frame_num
        )

        return video_clip, label

    def __len__(self):
        return len(self.dataset)
