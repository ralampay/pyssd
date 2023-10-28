import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.vgg16 import VGGBase

# Test VGGBase
tensors = torch.rand(1, 3, 300, 300)
vggbase = VGGBase()
conv4_3_feats, conv6_feats = vggbase(tensors)
expected_shape_conv4_3_feats = [1, 512, 38, 38]
expected_shape_conv6_feats = [1, 1024, 19, 19]
print(f"VGGBase forward() conv4_3_feats: {conv4_3_feats.shape}")
print(f"VGGBase forward() conv6_feats: {conv6_feats.shape}")

assert(expected_shape_conv4_3_feats == list(conv4_3_feats.shape))
assert(expected_shape_conv6_feats == list(conv6_feats.shape))
