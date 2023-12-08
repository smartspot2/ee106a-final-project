"""
Classify a rectified card image.

Used by vision.py
"""
from typing import Tuple, Union

import cv2
import numpy as np
import skimage as sk
import torch

from .labels import deserialize_label, label_to_string

CONFIDENCE_THRESH = 0.4


def classify(model, card) -> Union[str, None]:
    """
    Given an image (in cv2 BGR format) of a rectified card,
    classify the card using a trained neural net.
    """

    # convert into RGB; shape (280, 180, 3)
    card_rgb = sk.util.img_as_float(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))

    # move axis and convert into pytorch tensor
    card_input = torch.tensor(card_rgb).float().moveaxis((0, 1, 2), (1, 2, 0))

    pred_logits = model(card_input[None, ...])
    pred_argmax = torch.argmax(pred_logits)
    pred_max = pred_logits[pred_argmax]
    if pred_max < CONFIDENCE_THRESH:
        return None

    pred = int(pred_argmax.item())

    label = label_to_string(*deserialize_label(pred))
    return label


def classify_consensus(
    model, card_variations
) -> Union[Tuple[str, int], Tuple[None, None]]:
    """
    Given an array of images (each in cv2 BGR format) of a rectified card,
    classify the card using a trained neural net, and return the majority vote.
    """

    # convert into RGB; shape (280, 180, 3)
    cards_rgb = np.array(
        [
            sk.util.img_as_float(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
            for card in card_variations
        ]
    )

    # convert to tensor, move channels before width/heigth
    card_input = torch.tensor(cards_rgb).float().moveaxis((0, 1, 2, 3), (0, 2, 3, 1))

    pred_logits = model(card_input)
    pred_softmax = torch.nn.functional.softmax(pred_logits, dim=1)
    max_info = torch.max(pred_softmax, dim=1)
    max_indices = max_info.indices
    max_values = max_info.values

    if torch.all(max_values < CONFIDENCE_THRESH):
        return None, None

    majority_stats = torch.mode(max_indices)
    majority_pred = int(majority_stats.values.item())
    majority_idx = int(majority_stats.indices.item())

    label = label_to_string(*deserialize_label(majority_pred))
    return label, majority_idx
