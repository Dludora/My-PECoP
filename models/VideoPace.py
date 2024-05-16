import torch
import torch.nn as nn
from models.i3d_lora import I3D, Unit3Dpy


class VideoPace(nn.Module):
    def __init__(self, num_classes_p=5):
        super(VideoPace, self).__init__()

        self.num_classes_p = num_classes_p

        # x的每个位置的元素都有一定概率归零，模拟数据缺失，达到数据增强的目的。
        self.dropout = nn.Dropout(p=0.5)
        self.logits_p = Unit3Dpy(  # for the segment prediction head
            in_channels=1024,
            out_channels=num_classes_p,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False,
        )

        # loading the model and loading the pretraining weights - I3D with 3D-Adapters

        self.model = I3D(num_classes=400, dropout_prob=0.5)
        self.model.fix_i3d()

        ##############################################################################

    def load_pretrained_i3d(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)

    def forward(self, x):  ##########adding classification heads##############
        x = self.model(x)
        x_p = self.logits_p(self.dropout(x))

        l_p = x_p.squeeze(3).squeeze(3)
        l_p = torch.mean(l_p, 2)

        return l_p
