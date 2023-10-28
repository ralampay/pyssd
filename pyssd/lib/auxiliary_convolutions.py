from torch import nn

class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of base
        self.conv8_1 = nn.Conv2d(1024, 256, 1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, 1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.conv11_1 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, 3, padding=0)

        self.relu = nn.ReLU()

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                # Not working: nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, conv11_2
        """
        out = self.conv8_1(conv7_feats)             # (N, 256, 19, 19)
        out = self.relu(out)
        out = self.conv8_2(out)                     # (N, 512, 10, 10)
        out = self.relu(out)
        conv8_2_feats = out                         # (N, 512, 10, 10)

        out = self.conv9_1(out)                     # (N, 128, 10, 10)
        out = self.relu(out)
        out = self.conv9_2(out)                     # (N, 256, 5, 5)
        out = self.relu(out)
        conv9_2_feats = out                         # (N, 256, 5, 5)

        out = self.conv10_1(out)                    # (N, 128, 5, 5)
        out = self.relu(out)
        out = self.conv10_2(out)                    # (N, 256, 3, 3)
        out = self.relu(out)
        conv10_2_feats = out                        # (N, 256, 3, 3)

        out = self.conv11_1(out)                    # (N, 128, 3, 3)
        out = self.relu(out)
        conv11_2_feats = self.conv_11_2(out)        # (N, 256, 1, 1)
        conv11_2_feats = self.relu(conv11_2_feats)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
