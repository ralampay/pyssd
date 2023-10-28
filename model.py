from torch import nn
from math import sqrt
from itertools import product

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolution vgg16
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward propagation.

        :param x: image or tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = self.conv1_1(x)   # (N, 64, 300, 300)
        out = self.relu(out)
        out = self.conv1_2(out) # (N, 64, 300, 300)
        out = self.relu(out)
        out = self.pool1(out)   # (N, 64, 150, 150)

        out = self.conv2_1(out) # (N, 128, 150, 150)
        out = self.relu(out)
        out = self.conv2_2(out) # (N, 128, 150, 150)
        out = self.relu(out)
        out = self.pool2(out)   # (N, 128, 75, 75)

        out = self.conv3_1(out) # (N, 256, 75, 75)
        out = self.relu(out)
        out = self.conv3_2(out) # (N, 256, 75, 75)
        out = self.relu(out)
        out = self.conv3_3(out) # (N, 256, 75, 75)
        out = self.relu(out)
        out = self.pool3(out)   # (N, 256, 38, 38)

        out = self.conv4_1(out) # (N, 512, 38, 38)
        out = self.rlue(out)
        out = self.conv4_2(out) # (N, 512, 38, 38)
        out = self.relu(out)
        out = self.conv4_3(out) # (N, 512, 38, 38)
        out = self.relu(out)
        conv4_3_feats = out     # (N, 512, 38, 38)
        out = self.pool4(out)   # (N, 512, 19, 19)

        out = self.conv5_1(out) # (N, 512, 19, 19)
        out = self.relu(out)
        out = self.conv5_2(out) # (N, 512, 19, 19)
        out = self.relu(out)
        out = self.conv5_3(out) # (N, 512, 19, 19)
        out = self.relu(out)
        out = self.pool5(out)   # (N, 512, 19, 19) pool5 does not reduce dimensions

        out = self.conv6(out)   # (N, 1024, 19, 19)
        out = self.relu(out)

        conv7_feats = self.conv7(out)           # (N, 1024, 19, 19)
        conv7_feats = self.relu(conv7_feats)

        return conv4_3_feats, conv7_feats
