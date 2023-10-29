import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.vgg16 import VGGBase
from pyssd.lib.auxiliary_convolutions import AuxiliaryConvolutions
from pyssd.lib.prediction_convolutions import PredictionConvolutions

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
conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = aux_convs(conv7_feats)

expected_shape_a = [1, 512, 10, 10]
expected_shape_b = [1, 256, 5, 5]
expected_shape_c = [1, 256, 3, 3]
expected_shape_d = [1, 256, 1, 1]

print(f"AuxiliaryConvolutions forward() conv8_2_feats: {conv8_2_feats.shape}")
print(f"AuxiliaryConvolutions forward() conv9_2_feats: {conv9_2_feats.shape}")
print(f"AuxiliaryConvolutions forward() conv10_2_feats: {conv10_2_feats.shape}")
print(f"AuxiliaryConvolutions forward() conv11_2_feats: {conv11_2_feats.shape}")

assert(expected_shape_a == list(conv8_2_feats.shape))
assert(expected_shape_b == list(conv9_2_feats.shape))
assert(expected_shape_c == list(conv10_2_feats.shape))
assert(expected_shape_d == list(conv11_2_feats.shape))

num_classes = 2

pred_convs = PredictionConvolutions(num_classes)
locs, classes_scores = pred_convs(
    conv4_3_feats, 
    conv7_feats,
    conv8_2_feats,
    conv9_2_feats,
    conv10_2_feats,
    conv11_2_feats
)

expected_shape_locs = [1, 8732, 4]
expected_shape_classes_scores = [1, 8732, num_classes]

print(f"PredictionConvolutions forward() locs: {locs.shape}")
print(f"PredictionConvolutions forward() classes_scores: {classes_scores.shape}")

assert(expected_shape_locs == list(locs.shape))
assert(expected_shape_classes_scores == list(classes_scores.shape))
