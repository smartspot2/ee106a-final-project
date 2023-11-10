#!/usr/bin/env python

import os
import enum
import cv2
import skimage as sk
import matplotlib.pyplot as plt
from util.detect import detect_cards

plt.ion()


class CardShape(enum.Enum):
    SQUIGGLE = "squiggle"
    OVAL = "oval"
    DIAMOND = "diamond"


class CardColor(enum.Enum):
    RED = "red"
    GREEN = "green"
    PURPLE = "purple"


class CardNumber(enum.Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"


class CardShade(enum.Enum):
    SOLID = "solid"
    STRIPE = "stripe"
    OUTLINE = "outline"


LABEL_FORMAT = "{shape}-{color}-{number}-{shade}"


def print_options(enumClass):
    options = [e for e in enumClass]

    print(f"Choose {enumClass.__name__}:")
    for i, opt in enumerate(options, start=1):
        print(f"({i}) {opt.value}")
    while True:
        chosen = input("Enter a number: ")
        try:
            chosen = int(chosen) - 1
        except ValueError:
            continue

        break

    return options[chosen]


def main(args):
    plt.show()
    for file in os.listdir(args.image_dir):
        full_file = os.path.join(args.image_dir, file)
        if os.path.isfile(full_file):
            image = cv2.imread(full_file)
            cards = detect_cards(image)

            if len(cards) == 0:
                continue

            for card in cards:
                card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

                plt.imshow(card)
                skip = input("Press ENTER to label image, anything else to skip")
                if skip:
                    continue

                while True:
                    shape = print_options(CardShape)
                    color = print_options(CardColor)
                    number = print_options(CardNumber)
                    shade = print_options(CardShade)

                    label = LABEL_FORMAT.format(
                        shape=shape.value,
                        color=color.value,
                        number=number.value,
                        shade=shade.value,
                    )
                    print(f"Labeling as '{label}'")
                    again = input("Press ENTER to confirm, anything else to redo")
                    if not again:
                        break

                # save the card
                _, ext = os.path.splitext(file)
                nonce = None
                while True:
                    if nonce is None:
                        dest = os.path.join(args.out_dir, f"{label}{ext}")
                    else:
                        dest = os.path.join(args.out_dir, f"{label}_{nonce}{ext}")

                    if os.path.isfile(dest):
                        nonce = 1 if nonce is None else nonce + 1
                    else:
                        sk.io.imsave(dest, card)
                        break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("out_dir", help="Output directory for labeled images")

    main(parser.parse_args())
