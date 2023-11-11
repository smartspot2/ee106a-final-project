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
            nn.LazyConv2d(out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels=256, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.LazyLinear(out_features=81),
        )

    def forward(self, inp):
        """
        Forward pass of the classifier.

        Input shape should be (batch, channels, height, width).
        Output shape is (batch, 81).
        """
        return self.layers(inp)


class CardData(ImageFolder):
    """
    Subclass of torchvision.datasets.ImageFolder,
    with a custom class naming structure.
    """

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        labels = []
        for file in os.listdir(directory):
            full_file = os.path.join(directory, file)
            if not os.path.isfile(full_file):
                continue

            filename, _ = os.path.splitext(file)
            label = filename.split("_")[0]

            labels.append(label)

        # get the serialized form of each label
        label_mapping = {l: serialize_label(l) for l in labels}

        return labels, label_mapping

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
        for file in os.listdir(directory):
            full_file = os.path.join(directory, file)
            if not os.path.isfile(full_file):
                continue

            filename, _ = os.path.splitext(file)
            label = filename.split("_")[0]
            idx = class_to_idx[label]

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
