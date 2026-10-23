import sys, os, time, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

BASELINES_DIR = r"e:\IS403.R12-equity-return-predictability\src\training\baselines"
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)
import config
from tcn.tcn_model import TemporalConvNet

print("Device:", config.DEVICE)
# 5000 samples, lookback=20, features=16
x = torch.randn(5000, 20, 16)
y = torch.randn(5000)
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)

m = TemporalConvNet(num_inputs=16, num_channels=[64, 64, 64], kernel_size=3).to(config.DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
crit = nn.MSELoss()

t0 = time.time()
for ep in range(5):
    m.train()
    for bx, by in loader:
        bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
        opt.zero_grad()
        crit(m(bx), by).backward()
        opt.step()
    print(f"Epoch {ep+1} took {time.time() - t0:.2f}s")
    t0 = time.time()
