import torch
import torch.nn as nn
import torch.nn.functional as functional
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        out = self.conv2(functional.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, down=False):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.down = down

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        if self.down:
            out = functional.avg_pool1d(out, 2)
        return out


class ResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResidualUBlock, self).__init__()
        self.downsample = downsampling  # Flag to decide if down-sampling is needed
        K = 9  # Kernel size
        P = (K - 1) // 2  # Padding calculation

        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               padding=P,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()  # Encoder layers
        self.decoders = nn.ModuleList()  # Decoder layers

        # Creating encoder-decoder blocks
        for idx in range(layers):
            # Encoder block definition
            if idx == 0:
                # First encoder has different input channels
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=out_ch,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                # Subsequent encoders use mid_ch as input
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

            # Decoder block definition
            if idx == layers - 1:
                # Last decoder has different output channels
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=mid_ch * 2,
                        out_channels=out_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        output_padding=1,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                # Subsequent decoders output to mid_ch
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=mid_ch * 2,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        output_padding=1,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

        # Bottleneck layer (center of U-Net)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=K,
                padding=P,
                bias=False
            ),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        # Down-sampling layers (if required)
        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x):
        x_in = functional.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []
        for idx, layer in enumerate(self.encoders):
            # If output size is not divisible by 4, padding is added
            if out.size(-1) % 4 != 0:
                out = functional.pad(out, [1, 0, 0, 0])
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))

        # Trim the output to match the size of x_in (input)
        out = out[..., :x_in.size(-1)]

        out += x_in

        # If down-sampling is required, apply down-sampling layers
        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)


class ResU_Dense(nn.Module):
    def __init__(self, nOUT, in_ch=12, out_ch=256, mid_ch=64):
        super(ResU_Dense, self).__init__()
        # Initial convolutional layer
        self.conv = nn.Conv1d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=15,
                              padding=7,
                              stride=2,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

        # Define Residual U-blocks
        self.rub_0 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=6)
        self.rub_1 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=5)
        self.rub_2 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_3 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        # Parameters for dense blocks
        growthRate = 12
        reduction = 0.5
        nChannels = out_ch
        nDenseBlocks = 16

        # Define dense blocks and transitions
        self.dense1 = _make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = _make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        self.trans2 = Transition(nChannels, out_ch)

        # Multihead attention layer
        self.mha = nn.MultiheadAttention(out_ch, 8)
        # Max pooling layer to reduce dimensions
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layer
        self.fc_1 = nn.Linear(out_ch, nOUT)

    def forward(self, x):
        x = functional.leaky_relu(self.bn(self.conv(x)))

        # Pass through the residual U-blocks
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.rub_2(x)
        x = self.rub_3(x)

        # Pass through the dense blocks and transitions
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))

        # Apply dropout for regularization
        x = functional.dropout(x, p=0.5, training=self.training)

        x = x.permute(2, 0, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(1, 2, 0)

        # Reduce dimensions with pooling
        x = self.pool(x).squeeze(2)

        # Fully connected layer for final output
        x = self.fc_1(x)
        return x
