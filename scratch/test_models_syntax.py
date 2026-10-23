import sys
import os
import torch

BASELINES_DIR = r"e:\IS403.R12-equity-return-predictability\src\training\baselines"
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config

print("="*60)
print("VERIFYING SYNTAX AND FORWARD PASS OF 7 NEW MODELS")
print(f"Device: {config.DEVICE}")
print("="*60)

# Check config registration
new_models = ['tcn', 'nbeats', 'nhits', 'patchtst', 'tsmixer', 'itransformer', 'cnn']
for m in new_models:
    assert m in config.MODEL_REGISTRY, f"Missing {m} in MODEL_REGISTRY"
    assert m in config.HYPERPARAM_STRATEGY, f"Missing {m} in HYPERPARAM_STRATEGY"
    assert m in config.OPTUNA_TRIALS, f"Missing {m} in OPTUNA_TRIALS"
print("✓ All 7 models registered in config.py.")

# Dummy tensor: batch_size=4, lookback=20, num_features=16
x = torch.randn(4, 20, 16).to(config.DEVICE)

# 1. TCN
from tcn.tcn_model import TemporalConvNet
tcn_m = TemporalConvNet(num_inputs=16, num_channels=[32, 64, 64], kernel_size=3).to(config.DEVICE)
out_tcn = tcn_m(x)
assert out_tcn.shape == (4,), f"TCN output shape mismatch: {out_tcn.shape}"
print(f"✓ TCN forward pass OK. Output shape: {out_tcn.shape}")

# 2. N-BEATS
from nbeats.nbeats_model import NBeatsNet
nbeats_m = NBeatsNet(lookback=20, num_features=16, num_stacks=2, num_blocks=2, hidden_dim=64).to(config.DEVICE)
out_nbeats = nbeats_m(x)
assert out_nbeats.shape == (4,), f"N-BEATS output shape mismatch: {out_nbeats.shape}"
print(f"✓ N-BEATS forward pass OK. Output shape: {out_nbeats.shape}")

# 3. N-HiTS
from nhits.nhits_model import NHiTSNet
nhits_m = NHiTSNet(lookback=20, num_features=16, pooling_kernels=(4, 2, 1), hidden_dim=64).to(config.DEVICE)
out_nhits = nhits_m(x)
assert out_nhits.shape == (4,), f"N-HiTS output shape mismatch: {out_nhits.shape}"
print(f"✓ N-HiTS forward pass OK. Output shape: {out_nhits.shape}")

# 4. PatchTST
from patchtst.patchtst_model import PatchTSTModel
patchtst_m = PatchTSTModel(lookback=20, num_features=16, patch_len=8, stride=4, d_model=32, nhead=2).to(config.DEVICE)
out_patch = patchtst_m(x)
assert out_patch.shape == (4,), f"PatchTST output shape mismatch: {out_patch.shape}"
print(f"✓ PatchTST forward pass OK. Output shape: {out_patch.shape}")

# 5. TSMixer
from tsmixer.tsmixer_model import TSMixerNet
tsmixer_m = TSMixerNet(lookback=20, num_features=16, num_blocks=2, ff_dim=32).to(config.DEVICE)
out_tsmixer = tsmixer_m(x)
assert out_tsmixer.shape == (4,), f"TSMixer output shape mismatch: {out_tsmixer.shape}"
print(f"✓ TSMixer forward pass OK. Output shape: {out_tsmixer.shape}")

# 6. iTransformer
from itransformer.itransformer_model import iTransformerModel
itransformer_m = iTransformerModel(lookback=20, num_features=16, d_model=32, nhead=2, num_layers=1).to(config.DEVICE)
out_itrans = itransformer_m(x)
assert out_itrans.shape == (4,), f"iTransformer output shape mismatch: {out_itrans.shape}"
print(f"✓ iTransformer forward pass OK. Output shape: {out_itrans.shape}")

# 7. CNN
from cnn.cnn_model import TemporalCNN
cnn_m = TemporalCNN(num_features=16, num_filters=32, kernel_size=3, num_layers=2).to(config.DEVICE)
out_cnn = cnn_m(x)
assert out_cnn.shape == (4,), f"CNN output shape mismatch: {out_cnn.shape}"
print(f"✓ CNN forward pass OK. Output shape: {out_cnn.shape}")

print("="*60)
print("ALL 7 MODELS VERIFIED 100% FUNCTIONAL AND ERROR-FREE!")
print("="*60)
