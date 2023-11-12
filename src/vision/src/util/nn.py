#!/usr/bin/env python

"""
Neural network for classifying cards.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

from torch import nn
from torchvision.datasets import ImageFolder


class CardClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=256),
            nn.Flatten(),
            nn.Linear(in_features=768, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=81),
        )

    def forward(self, inp):
        """
        Forward pass of the classifier.

        Input shape should be (batch, channels, height, width).
        Output shape is (batch, 81).
        """
        return self.layers(inp)


class CardClassifierResnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.first_norm = nn.BatchNorm2d(num_features=64)
        self.first_relu = nn.ReLU(inplace=True)
        self.first_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            CardClassifierResnetBlock(in_channels=64, channels=64, downsample=False),
            CardClassifierResnetBlock(in_channels=64, channels=64, downsample=False),
        )
        self.block2 = nn.Sequential(
            CardClassifierResnetBlock(in_channels=64, channels=128, downsample=True),
            CardClassifierResnetBlock(in_channels=128, channels=128, downsample=False),
        )
        self.block3 = nn.Sequential(
            CardClassifierResnetBlock(in_channels=128, channels=256, downsample=True),
            CardClassifierResnetBlock(in_channels=256, channels=256, downsample=False),
        )

        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=81),
        )

    def forward(self, inp):
        """
        Forward pass of the classifier.

        Input shape should be (batch, channels, height, width).
        Output shape is (batch, 81)
        """

        out = self.first_conv(inp)
        out = self.first_norm(out)
        out = self.first_relu(out)
        out = self.first_maxpool(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.dense(out)

        return out


class CardClassifierResnetBlock(nn.Module):
    def __init__(self, in_channels, channels, downsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            # if downsampling, then set stride to 2
            stride=2 if downsample else 1,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(num_features=channels)
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(num_features=channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=1,
                    stride=2,
                ),
                nn.BatchNorm2d(channels),
            )
        else:
            self.downsample = None

    def forward(self, inp):
        identity = inp

        out = self.conv1(inp)  # potentially downsamples as well
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(inp)

        # add residual connection
        out += identity
        out = self.relu(out)

        return out


class CardData(ImageFolder):
    """
    Subclass of torchvision.datasets.ImageFolder,
    with a custom class naming structure.
    """

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        labels = set()
        for file in os.listdir(directory):
            full_file = os.path.join(directory, file)
            if not os.path.isfile(full_file):
                continue

            filename, _ = os.path.splitext(file)
            label = filename.split("_")[0]

            labels.add(label)

        # get the serialized form of each label
        label_mapping = {l: serialize_label(l) for l in labels}

        return list(labels), label_mapping

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        # not used
        del extensions, is_valid_file

        dataset = []
        file_list = []
        for file in os.listdir(directory):
            full_file = os.path.join(directory, file)
            if not os.path.isfile(full_file):
                continue
            file_list.append(file)

        file_list.sort()

        for file in file_list:
            filename, _ = os.path.splitext(file)
            label = filename.split("_")[0]
            idx = class_to_idx[label]

            full_file = os.path.join(directory, file)
            dataset.append((full_file, idx))

        return dataset


SHAPES = ["oval", "squiggle", "diamond"]
COLORS = ["red", "purple", "green"]
COUNTS = ["1", "2", "3"]
SHADES = ["solid", "stripe", "outline"]


def serialize_label(label: str) -> int:
    """
    Serialize a card label from a string.

    Label must be of the form:
        "{shape}-{color}-{count}-{shade}"
    If duplicates, may have suffix "_{nonce}"

    Serialization is a number in base 3 with the digits
        (((shape) * 3 + color) * 3 + count) * 3 + shade
    """
    label = label.split("_")[0]
    shape, color, count, shade = label.split("-")
    shape_int = SHAPES.index(shape)
    color_int = COLORS.index(color)
    count_int = COUNTS.index(count)
    shade_int = SHADES.index(shade)

    return shape_int * 27 + color_int * 9 + count_int * 3 + shade_int


def deserialize_label(label: int) -> tuple[int, int, int, int]:
    """
    Deserialize a label from an int.

    label % 3 => shade
    (label // 3) % 3 => count
    (label // 9) % 3 => color
    (label // 27) % 3 => shape
    """

    shade = label % 3
    count = (label // 3) % 3
    color = (label // 9) % 3
    shape = (label // 27) % 3

    return shape, color, count, shade


def label_to_string(shape: int, color: int, count: int, shade: int) -> str:
    shape_str = SHAPES[shape]
    color_str = COLORS[color]
    count_str = COUNTS[count]
    shade_str = SHADES[shade]

    return f"{shape_str}-{color_str}-{count_str}-{shade_str}"
