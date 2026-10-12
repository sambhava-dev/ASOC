"""Compact ResNet-style anti-spoofing network over log-mel input (needs PyTorch)."""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(s, i, o, stride):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.sk = None if (i == o and stride == 1) else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        y = s.b2(s.c2(F.relu(s.b1(s.c1(x)))))
        return F.relu(y + (x if s.sk is None else s.sk(x)))


class SpoofNet(nn.Module):
    """Input (B,1,n_mels,T). Output logits (B,n_classes); class 0 is bona fide, 1.. are spoof classes."""
    def __init__(s, n_classes=2, width=32):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        s.body = nn.Sequential(Block(width, width, 1), Block(width, 2 * width, 2), Block(2 * width, 4 * width, 2), Block(4 * width, 8 * width, 2))
        s.drop = nn.Dropout(.3); s.fc = nn.Linear(16 * width, n_classes)

    def forward(s, x):
        x = s.body(s.stem(x))
        return s.fc(s.drop(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))
