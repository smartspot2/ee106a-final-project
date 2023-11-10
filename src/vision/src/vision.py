#!/usr/bin/env python

"""
Vision node to detect Set cards.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from util.detect import detect_cards, find_contours


def main(args):
    image = cv2.imread(args.image)

    # find contours (for display)
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
    cards = detect_cards(image)

    # display
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(output)

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

    main(parser.parse_args())
