
import torch
import torch.nn as nn
from einops import rearrange


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, num_heads=8, dim=96, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(inplace=True))

        self.trans = nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads, dim_feedforward=dim, dropout=dropout)
        self.trans_encoder = nn.TransformerEncoder(self.trans, num_layers=num_layers)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.interpolate(x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
        x = rearrange(x, 'b c w -> w b c')
        x = self.trans_encoder(x)
        x = rearrange(x, 'w b c -> b c w')
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest", recompute_scale_factor=True)
        return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.dec4 = DecoderBlock(512, 512)
        self.dec3 = DecoderBlock(256 + 512, 256)
        self.dec2 = DecoderBlock(128 + 256, 128)
        self.dec1 = DecoderBlock(64 + 128, 64)

        self.head = nn.Conv1d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # print('e1:', enc1.shape)
        # print('e2:', enc2.shape)
        # print('e3:', enc3.shape)
        # print('e4:', enc4.shape)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat((enc3, dec4), dim=1))
        dec2 = self.dec2(torch.cat((enc2, dec3), dim=1))
        dec1 = self.dec1(torch.cat((enc1, dec2), dim=1))

        return self.head(dec1)
