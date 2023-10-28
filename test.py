import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.vgg16 import VGGBase
from pyssd.lib.auxiliary_convolutions import AuxiliaryConvolutions

# Test VGGBase
tensors = torch.rand(1, 3, 300, 300)
vggbase = VGGBase()
conv4_3_feats, conv7_feats = vggbase(tensors)

expected_shape_conv4_3_feats = [1, 512, 38, 38]
expected_shape_conv7_feats = [1, 1024, 19, 19]

print(f"VGGBase forward() conv4_3_feats: {conv4_3_feats.shape}")
print(f"VGGBase forward() conv6_feats: {conv7_feats.shape}")

assert(expected_shape_conv4_3_feats == list(conv4_3_feats.shape))
assert(expected_shape_conv7_feats == list(conv7_feats.shape))

aux_convs = AuxiliaryConvolutions()
a, b, c, d = aux_convs(conv7_feats)

expected_shape_a = [1, 512, 10, 10]
expected_shape_b = [1, 256, 5, 5]
expected_shape_c = [1, 256, 3, 3]
expected_shape_d = [1, 256, 1, 1]

print(f"AuxiliaryConvolutions forward() conv8_2_feats: {a.shape}")
print(f"AuxiliaryConvolutions forward() conv9_2_feats: {b.shape}")
print(f"AuxiliaryConvolutions forward() conv10_2_feats: {c.shape}")
print(f"AuxiliaryConvolutions forward() conv11_2_feats: {d.shape}")

assert(expected_shape_a == list(a.shape))
assert(expected_shape_b == list(b.shape))
assert(expected_shape_c == list(c.shape))
assert(expected_shape_d == list(d.shape))
