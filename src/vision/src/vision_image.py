#!/usr/bin/env python

"""
Vision node to detect Set cards.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import torch
from util.classify import classify_consensus
from util.detect import detect_cards, find_contours
from util.nn import CardClassifier, CardClassifierResnet
from util.set import find_set


def load_model(model_file, use_resnet=False):
    if model_file is None:
        return None

    if use_resnet:
        model = CardClassifierResnet()
    else:
        model = CardClassifier()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    return model


def main_detect(image, model=None):
    # find contours
    contours = find_contours(image)

    # find cards
    cards = detect_cards(image, contours, variations=5)

    if model is not None:
        # classify cards
        labels = []
        # single_cards = []
        for card_variations in cards:
            label, majority_idx = classify_consensus(model, card_variations)
            labels.append(label)
            # single_cards.append(card_variations[majority_idx])
        # cards = single_cards
        card_shape = cards[0][0].shape
        cards = np.array(cards).reshape(-1, *card_shape)

        # find the set
        found_set = find_set(labels)
    else:
        labels = [""] * len(contours)
        found_set = None

    # ===== display =====

    # draw contours
    output = image.copy()
    for idx, (card, contour) in enumerate(zip(cards, contours)):
        if found_set is not None and idx in found_set:
            # part of set
            rgb = (0, 255, 0)
        else:
            # not part of set
            rgb = (255, 0, 0)

        output = cv2.drawContours(output, contours, idx, rgb, 2, cv2.LINE_8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, width_ratios=(2, 1))
    ax1.imshow(output)
    for contour, label in zip(contours, labels):
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        ax1.text(
            cx, cy, label.replace("-", "\n"), fontsize="small", ha="center", va="center"
        )

    output_cards = cv2.cvtColor(
        sk.util.montage(
            cards,
            channel_axis=-1,
            fill=[0, 0, 0],
            rescale_intensity=True,
        ),
        cv2.COLOR_BGR2RGB,
    )
    ax2.imshow(output_cards)
    plt.show()


def main(args):
    image = cv2.imread(args.image)

    model = load_model(args.model, use_resnet=args.use_resnet)

    # detect once and exit
    main_detect(image, model=model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Input image")

    parser.add_argument(
        "--model",
        default=None,
        help="Model file to load for use as the card classifier",
    )
    parser.add_argument("--use-resnet", action="store_true")

    main(parser.parse_args())
