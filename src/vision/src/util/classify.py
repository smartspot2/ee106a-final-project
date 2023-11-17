"""
Classify a rectified card image.

Used by vision.py
"""
import cv2
import skimage as sk
import torch

from .labels import deserialize_label, label_to_string


def classify(model, card) -> str:
    """
    Given an image (in cv2 BGR format) of a rectified card,
    classify the card using a trained neural net.
    """

    # convert into RGB; shape (280, 180, 3)
    card_rgb = sk.util.img_as_float(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))

    # move axis and convert into pytorch tensor
    card_input = torch.tensor(card_rgb).float().moveaxis((0, 1, 2), (1, 2, 0))

    pred_logits = model(card_input[None, ...])
    pred = torch.argmax(pred_logits)
    pred = pred.item()

    label = label_to_string(*deserialize_label(pred))
    return label
