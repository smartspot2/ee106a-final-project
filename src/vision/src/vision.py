#!/usr/bin/env python

"""
Vision node to detect Set cards.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import torch
from util.classify import classify
from util.detect import detect_cards, find_contours
from util.nn import CardClassifier


def main(args):
    image = cv2.imread(args.image)

    # find contours
    contours = find_contours(image)

    output = image.copy()
    for idx in range(len(contours)):
        rgb = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        output = cv2.drawContours(output, contours, idx, rgb, 2, cv2.LINE_8)

    # find cards
    cards = detect_cards(image, contours)

    # classify cards
    model = CardClassifier()
    model.load_state_dict(torch.load(args.classifier_model))

    labels = []
    for card in cards:
        label = classify(model, card)
        labels.append(label)

    # display
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(output)
    for contour, label in zip(contours, labels):
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        plt.text(cx, cy, label.replace("-", "\n"), fontsize="small", ha="center", va="center")

    output_cards = cv2.cvtColor(
        sk.util.montage(cards, channel_axis=-1, fill=[0, 0, 0], rescale_intensity=True),
        cv2.COLOR_BGR2RGB,
    )
    plt.subplot(122)
    plt.imshow(output_cards)
    # plt.imshow(rectified_cards[1])
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Input image")
    parser.add_argument(
        "classifier_model", help="Model file to load for use as the card classifier"
    )

    main(parser.parse_args())
