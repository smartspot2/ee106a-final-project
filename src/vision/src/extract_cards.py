#!/usr/bin/env python

import enum
import os

import cv2
import matplotlib.pyplot as plt
import skimage as sk
from util.detect import detect_cards, find_contours

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
    all_cards = []
    files = sorted(os.listdir(args.image_dir))
    for file in files:
        full_file = os.path.join(args.image_dir, file)
        if os.path.isfile(full_file):
            image = cv2.imread(full_file)
            contours = find_contours(image)
            cards = detect_cards(image, contours, preprocess=False)

            if len(cards) == 0:
                continue

            for card in cards:
                card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

                all_cards.append((card, full_file))

    if args.by_attribute:
        # classify by attribute instead of by card
        labels = [tuple() for _ in range(len(all_cards))]
        for card_enum in (CardShape, CardColor, CardNumber, CardShade):
            plt.clf()
            print(f"===== CLASSIFYING {card_enum.__name__} =====")
            input("Press ENTER to continue.")
            for idx, (card, full_file) in enumerate(all_cards):
                plt.clf()
                plt.imshow(card)
                skip = input(
                    f"[{idx+1}/{len(all_cards)}] Press ENTER to label image, anything else to skip"
                )
                if skip:
                    continue

                selected = print_options(card_enum)
                labels[idx] = (*labels[idx], selected.value)

        # save all as files
        for (card, full_file), label_values in zip(all_cards, labels):
            label = LABEL_FORMAT.format(
                shape=label_values[0],
                color=label_values[1],
                number=label_values[2],
                shade=label_values[3],
            )

            # save the card
            file = os.path.basename(full_file)
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

    else:
        for idx, (card, full_file) in enumerate(all_cards):
            file = os.path.basename(full_file)

            plt.clf()
            plt.imshow(card)
            skip = input(
                f"[{idx+1}/{len(all_cards)}] Press ENTER to label image, anything else to skip"
            )
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
    parser.add_argument("--by-attribute", action="store_true")

    main(parser.parse_args())
