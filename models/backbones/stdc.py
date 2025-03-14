import math

import torch
import torch.nn as nn
from torch.nn import init


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias=False,
                ),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 2, stride=stride)
                )
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 4, stride=stride)
                )
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 2, stride=stride)
                )
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 4, stride=stride)
                )
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class STDCNet(nn.Module):
    def __init__(
        self,
        base=64,
        layers=[2, 2, 2],
        block_num=4,
        block_type="cat",
        use_conv_last=False,
    ):
        super().__init__()
        if block_type == "cat":
            block = CatBottleneck
        elif block_type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)

        if layers != [2, 2, 2] and layers != [4, 5, 3]:
            layers = [4, 5, 3]
        if layers == [2, 2, 2]:
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])
        elif layers == [4, 5, 3]:
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])

        if self.use_conv_last:
            self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 1)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            2,
                        )
                    )
                else:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 2)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            1,
                        )
                    )

        return nn.Sequential(*features)

    def forward(self, x):
        outs = {}
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        outs["res2"] = feat4

        feat8 = self.x8(feat4)
        outs["res3"] = feat8

        feat16 = self.x16(feat8)
        outs["res4"] = feat16

        feat32 = self.x32(feat16)
        outs["res5"] = feat32

        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
            outs["res5"] = feat32
        return outs
    
    def forward_specific_stage(self, x, start_layer):
        """
        Forward the STDCNet model starting from a specific stage.

        Args:
            x (Tensor):
                - If start_layer == "res2", x is expected to be a raw input image tensor of shape (N, C, H, W).
                - Otherwise (for "res3", "res4", or "res5"), x is expected to be the feature map
                  computed from the preceding stage.
            start_layer (str): Indicates the stage from which to begin forwarding.
                Must be one of: "res2", "res3", "res4", "res5".

        Returns:
            dict[str, Tensor]: A dictionary with the computed features starting from the specified stage.
                Keys are the same as in the full forward pass (e.g., "res2", "res3", "res4", "res5").
        """
        outs = {}
        if start_layer == "res2":
            # For "res2", compute from raw input.
            feat2 = self.x2(x)
            feat4 = self.x4(feat2)
            outs["res2"] = feat4

            feat8 = self.x8(feat4)
            outs["res3"] = feat8

            feat16 = self.x16(feat8)
            outs["res4"] = feat16

            feat32 = self.x32(feat16)
            if self.use_conv_last:
                feat32 = self.conv_last(feat32)
            outs["res5"] = feat32

        elif start_layer == "res3":
            # For "res3", assume x is already the output of the previous block (i.e. "res2").
            feat8 = self.x8(x)
            outs["res3"] = feat8

            feat16 = self.x16(feat8)
            outs["res4"] = feat16

            feat32 = self.x32(feat16)
            if self.use_conv_last:
                feat32 = self.conv_last(feat32)
            outs["res5"] = feat32

        elif start_layer == "res4":
            # For "res4", assume x is already the output of "res3".
            feat16 = self.x16(x)
            outs["res4"] = feat16

            feat32 = self.x32(feat16)
            if self.use_conv_last:
                feat32 = self.conv_last(feat32)
            outs["res5"] = feat32

        elif start_layer == "res5":
            # For "res5", assume x is already the output of "res4".
            feat32 = self.x32(x)
            if self.use_conv_last:
                feat32 = self.conv_last(feat32)
            outs["res5"] = feat32

        else:
            raise ValueError("Invalid start_layer. Must be one of 'res2', 'res3', 'res4', or 'res5'.")

        return outs

