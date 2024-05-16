import os
import torch
import random
import scipy.io
from torch.utils.data import Dataset
from utils import misc
from PIL import Image
import ipdb


class Seven_Dataset(Dataset):
    """AQA-7 dataset"""

    def __init__(self, args, transforms, color_jitter, subset):
        self.subset = subset
        self.transforms = transforms

        classes_name = [
            "diving",
            "gym_vault",
            "ski_big_air",
            "snowboard_big_air",
            "sync_diving_3m",
            "sync_diving_10m",
        ]
        self.sport_class = classes_name[args.class_idx - 1]

        self.class_idx = args.class_idx
        # file path
        self.data_root = args.data_root
        self.data_path = os.path.join(self.data_root, "{}-out".format(self.sport_class))
        self.split_path = os.path.join(
            self.data_root, "Split_4", "split_4_{}_list.mat".format(self.subset)
        )
        self.split = scipy.io.loadmat(self.split_path)[
            "consolidated_{}_list".format(self.subset)
        ]
        self.split = self.split[self.split[:, 0] == self.class_idx].tolist()
        self.dataset = self.split.copy()

        # setting
        self.length = args.frame_length

        # VSPP
        self.color_jitter = color_jitter
        self.clip_len = args.clip_len
        self.max_sample_rate = args.max_sample_rate
        self.max_segment = args.max_segment
        self.fr = args.fr

    def __len__(self):
        return len(self.dataset)

    def load_video(self, index, sample_rate, segment_start_frame, segment_last_frame):
        video_path = os.path.join(self.data_path, "%03d" % index)

        video_clip = []
        idx = 1
        idx1 = 1

        while idx <= self.length and idx1 <= self.clip_len:
            cur_img_path = os.path.join(video_path, "img_%05d.jpg" % idx)
            idx += (
                sample_rate
                if segment_start_frame <= idx1 <= segment_last_frame
                else self.fr
            )
            idx1 += 1
            img = Image.open(cur_img_path)
            video_clip.append(img)

        if self.subset == "train":
            video_clip = [self.color_jitter(img) for img in video_clip]

        video_clip = self.transforms(video_clip)

        return video_clip

    def __getitem__(self, index):
        sample = self.dataset[index]
        assert int(sample[0]) == self.class_idx
        idx = int(sample[1])

        sample_rate = random.randint(1, self.max_sample_rate)
        while sample_rate == self.fr:
            sample_rate = random.randint(1, self.max_sample_rate)

        # segment of the random clip
        segment = random.randint(1, self.max_segment)

        # 选中的segment的起始帧和结束帧
        segment_start_frame = int((segment - 1) * (self.clip_len / self.max_segment))
        segment_last_frame = int(segment * (self.clip_len / self.max_segment))

        video_clip = self.load_video(
            idx, sample_rate, segment_start_frame, segment_last_frame
        )

        label_speed = sample_rate - 1
        label_segment = segment - 1

        return video_clip, torch.tensor([label_speed, label_segment])
