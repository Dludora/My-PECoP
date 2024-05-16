import logging
import os
import yaml
import shutil
from torchvideotransforms import video_transforms, volume_transforms
from torchvision import transforms
from utils import misc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class Processor(object):
    def __init__(self, args):
        self.args = args
        self.init_log()
        self.save_config()
        self.load_data()
        self.load_model()
        self.load_loss()
        self.get_optimizer_scheduler()
        self.load_pretrain()

    def init_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # stream handler
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

        # file handler
        log_dir = os.path.join(self.args.exp_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{self.args.subset}.log")
        if (
            self.args.resume and self.args.subset == "train"
        ) or self.args.subset == "test":
            file_handler = logging.FileHandler(log_file, mode="a")
        else:
            file_handler = logging.FileHandler(log_file, mode="w")

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        self.logger = logger

    def save_config(self):
        # save config file
        args_dict = vars(self.args)
        config_dir = os.path.join(self.args.exp_path, "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, f"{self.args.benchmark}.yaml")
        with open(config_file, "w") as f:
            yaml.dump(args_dict, f)
            self.logger.info(f"Save config file to {config_file}.")

        # save model file
        if self.args.subset == "train":
            model_dir = os.path.join(self.args.exp_path, "models")
            if self.args.subset == "train":
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                shutil.copytree("./models", model_dir)

                self.logger.info(f"Save model file to {model_dir}.")

    def load_data(self):
        self.logger.info("Load data.")
        train_trans, color_jitter = self.get_transforms()
        self.dataloader = {}
        Dataset = misc.import_class(
            f"datasets.{self.args.benchmark}.{self.args.benchmark}_Dataset"
        )
        train_dataset = Dataset(
            args=self.args,
            transforms=train_trans,
            color_jitter=color_jitter,
            subset="train",
        )
        if self.args.subset == "train":
            self.dataloader["train"] = DataLoader(
                train_dataset,
                batch_size=self.args.bs_train,
                shuffle=True,
                pin_memory=True,
                num_workers=int(self.args.num_workers),
            )

        test_trans, color_jitter = self.get_transforms(subset="test")
        test_dataset = Dataset(
            args=self.args,
            transforms=test_trans,
            color_jitter=color_jitter,
            subset="test",
        )
        self.dataloader["test"] = DataLoader(
            test_dataset,
            batch_size=self.args.bs_test,
            shuffle=False,
            pin_memory=True,
            num_workers=int(self.args.num_workers),
        )

    def get_transforms(self, subset="train"):
        self.logger.info(f"Get {self.args.subset} transforms.")
        if subset == "train":
            trans = video_transforms.Compose(
                [
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.Resize((455, 256)),
                    video_transforms.RandomCrop(224),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            trans = video_transforms.Compose(
                [
                    video_transforms.Resize((455, 256)),
                    video_transforms.CenterCrop(224),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        color_jitter = transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )
        color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        return trans, color_jitter

    def load_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.logger.info("Load model.")
        Model = misc.import_class(f"models.VideoPace.VideoPace")
        if self.args.benchmark == "MTL_Pace":
            self.base_model = (
                Model(
                    num_classes_p=self.args.max_sample_rate,
                ).cuda()
                if torch.cuda.is_available()
                else Model(
                    num_classes_p=self.args.max_sample_rate,
                )
            )
        else:
            self.base_model = (
                Model(
                    num_classes_p=self.args.max_sample_rate,
                    num_classes_s=self.args.max_segment,
                ).cuda()
                if torch.cuda.is_available()
                else Model(
                    num_classes_p=self.args.max_sample_rate,
                    num_classes_s=self.args.max_segment,
                )
            )
        self.logger.info(f"{len(self.args.gpu.split(','))} GPUs are used.")

    def load_loss(self):
        self.logger.info("Load loss.")
        self.criterion = (
            nn.CrossEntropyLoss().cuda()
            if torch.cuda.is_available()
            else nn.CrossEntropyLoss()
        )

    def get_optimizer_scheduler(self):
        self.logger.info("Get optimizer and scheduler.")
        self.optimizer = optim.SGD(
            self.base_model.parameters(),
            lr=self.args.learn_rate,
            momentum=self.args.momentum,
            weight_decay=float(self.args.weight_decay),
            dampening=self.args.dampening,
            nesterov=self.args.nesterov,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.T_max,
            eta_min=self.args.learn_rate / 1000,
        )

    def load_pretrain(self):
        self.history = {}
        self.history["train"] = []
        self.history["test"] = []

        if not self.args.resume and self.args.subset == "train":
            if not os.path.exists(self.args.ckpts):
                self.logger.info(f"Checkpoint file {self.args.ckpts} not exists.")
                raise FileNotFoundError

            self.logger.info(f"Load checkpoint file {self.args.ckpts}.")
            self.base_model.load_pretrained_i3d(self.args.ckpts)
            # initial params
            self.start_epoch = 0
            self.epoch_best = 0
            self.avg_best = 0
            self.loss_best = 0

        else:
            self.args.ckpts = (
                os.path.join(self.args.exp_path, "weights/best.pth")
                if self.args.subset == "test"
                else os.path.join(self.args.exp_path, "weights/last.pth")
            )
            if not os.path.exists(self.args.ckpts):
                self.logger.info(f"Checkpoint file {self.args.ckpts} not exists.")
                raise FileNotFoundError
            self.logger.info(f"Load checkpoint file {self.args.ckpts}.")
            state_dict = torch.load(self.args.ckpts)
            self.base_model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])

            # initial params
            self.start_epoch = state_dict["epoch"] + 1
            self.epoch_best = state_dict["epoch_best"]
            self.avg_best = state_dict["avg_best"]
            self.loss_best = state_dict["loss_best"]

        # 设置多卡
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.base_model = nn.DataParallel(self.base_model)

    def save_checkpoint(self, epoch, best=False):
        if best:
            ckpt_file = os.path.join(self.args.exp_path, "weights/best.pth")
            self.epoch_best = epoch
            self.avg_best = self.history["train"][-1][1]
            self.loss_best = self.history["train"][-1][0]
        else:
            ckpt_file = os.path.join(self.args.exp_path, "weights/last.pth")

        if not os.path.exists(os.path.dirname(ckpt_file)):
            os.makedirs(os.path.dirname(ckpt_file))

        # 判断是否使用了多卡
        model = (
            self.base_model.module
            if isinstance(self.base_model, nn.DataParallel)
            else self.base_model
        )
        state_dict = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "epoch_best": self.epoch_best,
            "avg_best": self.avg_best,
            "loss_best": self.loss_best,
        }
        torch.save(state_dict, ckpt_file)

    def train_pace(self, epoch):
        total_loss = 0.0
        correct = 0
        it = 0

        self.base_model.train()
        loader = self.dataloader["train"]
        process = tqdm(loader, dynamic_ncols=True)
        for idx, sample in enumerate(process):
            rgb_clip, label_speed = sample
            rgb_clip = rgb_clip.to(dtype=torch.float).cuda()
            label_speed = label_speed.cuda()
            
            self.optimizer.zero_grad()
            out = self.base_model(rgb_clip)
            loss = self.criterion(out, label_speed)

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            probs_speed = nn.Softmax(dim=1)(out)
            preds_speed = torch.max(probs_speed, 1)[1]
            accuracy_speed = (
                torch.sum(preds_speed == label_speed.data)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            accuracy = (accuracy_speed / 2) / self.args.bs_train
            correct += accuracy
            it += 1

            process.set_description(
                f"Epoch {epoch+1} | Loss: {loss.item()} | Acc: {accuracy}"
            )
            process.update()

        process.close()

        # analyze the results
        avg_loss = total_loss / it
        avg_acc = correct / it
        self.history["train"].append((avg_loss, avg_acc))
        self.logger.info(
            f"Train: Epoch {epoch+1} | Loss: {avg_loss} | Acc: {avg_acc} | lr: {self.scheduler.get_last_lr()}"
        )

        if self.scheduler is not None:
            self.scheduler.step()

    def eval_pace(self, epoch):
        self.base_model.eval()
        loader = self.dataloader["test"]

        total_loss = 0.0
        correct = 0
        it = 0

        process = tqdm(loader, dynamic_ncols=True)
        with torch.no_grad():
            for idx, sample in enumerate(process):
                rgb_clip, label = sample
                rgb_clip = rgb_clip.to(dtype=torch.float).cuda()
                label_speed = label.cuda()

                out = self.base_model(rgb_clip)
                loss = self.criterion(out, label_speed)

                total_loss += loss.item()

                probs_speed = nn.Softmax(dim=1)(out)
                preds_speed = torch.max(probs_speed, 1)[1]
                accuracy_speed = (
                    torch.sum(preds_speed == label_speed.data)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                accuracy = accuracy_speed / self.args.bs_test
                correct += accuracy
                it += 1

                process.set_description(f"Test | Loss: {loss.item()} | Acc: {accuracy}")
                process.update()

        process.close()

        avg_loss = total_loss / it
        avg_acc = correct / it
        self.history["test"].append((avg_loss, avg_acc))
        self.logger.info(f"Test: Loss: {avg_loss} | Acc: {avg_acc}")

        best = avg_acc > self.avg_best
        self.save_checkpoint(epoch, best=best)

    def train(self, epoch):
        total_loss = 0.0
        correct = 0
        it = 0

        self.base_model.train()
        loader = self.dataloader["train"]
        process = tqdm(loader, dynamic_ncols=True)
        for idx, sample in enumerate(process):
            rgb_clip, labels = sample
            rgb_clip = rgb_clip.to(dtype=torch.float).cuda()
            label_speed = labels[:, 0].cuda()
            label_segment = labels[:, 1].cuda()

            self.optimizer.zero_grad()
            out1, out2 = self.base_model(rgb_clip)
            loss1 = self.criterion(out1, label_speed)
            loss2 = self.criterion(out2, label_segment)
            loss = loss1 + loss2

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            probs_segment = nn.Softmax(dim=1)(out2)
            preds_segment = torch.max(probs_segment, 1)[1]
            accuracy_seg = (
                torch.sum(preds_segment == label_segment.data)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            probs_speed = nn.Softmax(dim=1)(out1)
            preds_speed = torch.max(probs_speed, 1)[1]
            accuracy_speed = (
                torch.sum(preds_speed == label_speed.data)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            accuracy = ((accuracy_speed + accuracy_seg) / 2) / self.args.bs_train
            correct += accuracy
            it += 1

            process.set_description(
                f"Epoch {epoch+1} | Loss: {loss.item()} | Acc: {accuracy}"
            )
            process.update()

        process.close()

        # analyze the results
        avg_loss = total_loss / it
        avg_acc = correct / it
        self.history["train"].append((avg_loss, avg_acc))
        self.logger.info(
            f"Train: Epoch {epoch+1} | Loss: {avg_loss} | Acc: {avg_acc} | lr: {self.scheduler.get_last_lr()}"
        )

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, epoch):
        self.base_model.eval()
        loader = self.dataloader["test"]

        total_loss = 0.0
        correct = 0
        it = 0

        process = tqdm(loader, dynamic_ncols=True)
        with torch.no_grad():
            for idx, sample in enumerate(process):
                rgb_clip, labels = sample
                rgb_clip = rgb_clip.to(dtype=torch.float).cuda()
                label_speed = labels[:, 0].cuda()
                label_segment = labels[:, 1].cuda()

                out1, out2 = self.base_model(rgb_clip)
                loss1 = self.criterion(out1, label_speed)
                loss2 = self.criterion(out2, label_segment)
                loss = loss1 + loss2

                total_loss += loss.item()

                probs_segment = nn.Softmax(dim=1)(out2)
                preds_segment = torch.max(probs_segment, 1)[1]
                accuracy_seg = (
                    torch.sum(preds_segment == label_segment.data)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                probs_speed = nn.Softmax(dim=1)(out1)
                preds_speed = torch.max(probs_speed, 1)[1]
                accuracy_speed = (
                    torch.sum(preds_speed == label_speed.data)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                accuracy = ((accuracy_speed + accuracy_seg) / 2) / self.args.bs_test
                correct += accuracy
                it += 1

                process.set_description(f"Test | Loss: {loss.item()} | Acc: {accuracy}")
                process.update()

        process.close()

        avg_loss = total_loss / it
        avg_acc = correct / it
        self.history["test"].append((avg_loss, avg_acc))
        self.logger.info(f"Test: Loss: {avg_loss} | Acc: {avg_acc}")

        best = avg_acc > self.avg_best
        self.save_checkpoint(epoch, best=best)

    def start(self):
        model_params = misc.count_params(self.base_model)
        self.logger.info(f"Model has {model_params} parameters.")

        if self.args.subset == "train":
            for epoch in range(self.args.epoch):
                self.logger.info(
                    f"+--------------------------------------"
                    f"[ Epoch {epoch} start. ]"
                    f"--------------------------------------+"
                )
                self.train_pace(epoch)
                self.eval_pace(epoch)
                self.logger.info(
                    f"+--------------------------------------"
                    f"[ Epoch {epoch} end. ]"
                    f"--------------------------------------+"
                )
        else:
            self.eval_pace(0)
