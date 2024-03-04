import torch
import torch.nn as nn
from models.i3d_for_USDL import InceptionI3d, Unit3D
from models.i3d_lora import I3D, Unit3Dpy


class VSPP(nn.Module):
    def __init__(self, num_classes_p=5, num_classes_s=4):
        super(VSPP, self).__init__()

        self.num_classes_s = num_classes_s
        self.num_classes_p = num_classes_p

        # x的每个位置的元素都有一定概率归零，模拟数据缺失，达到数据增强的目的。
        self.dropout = nn.Dropout(p=0.5)
        self.logits_p = Unit3Dpy(  # for the segment prediction head
            in_channels=1024,
            out_channels=num_classes_p,
            kernel_size=(3, 3, 3),
            activation=None,
            use_bias=True,
            use_bn=False,
        )

        self.logits_s = Unit3Dpy(  # for the video playback prediction head
            in_channels=1024,
            out_channels=self.num_classes_s,
            kernel_size=(3, 3, 3),
            activation=None,
            use_bias=True,
            use_bn=False,
        )

        # loading the model and loading the pretraining weights - I3D with 3D-Adapters

        self.model = InceptionI3d()
        self.model.fix_i3d()

        ##############################################################################

    def load_pretrained_i3d(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)

    def forward(self, x):  ##########adding classification heads##############
        x = self.model(x)
        x_p = self.logits_p(self.dropout(x))
        x_s = self.logits_s(self.dropout(x))

        l_p = x_p.squeeze(3).squeeze(3)
        l_s = x_s.squeeze(3).squeeze(3)
        l_p = torch.mean(l_p, 2)
        l_s = torch.mean(l_s, 2)

        return l_p, l_s
