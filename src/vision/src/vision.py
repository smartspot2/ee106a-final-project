#!/usr/bin/env python

"""
Vision node to detect Set cards.
"""

import io
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import rospy
import time
from functools import partial
import torch
from util.classify import classify
from util.detect import detect_cards, find_contours
from util.nn import CardClassifier, CardClassifierResnet
from util.set import find_set

from sensor_msgs.msg import Image
import cv_bridge

matplotlib.use("Agg")


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


def main_detect(image, model=None, fig=None, ax=None):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    # find contours
    contours = find_contours(image)

    # find cards
    cards = detect_cards(image, contours)

    if model is not None:
        # classify cards
        labels = []
        for card in cards:
            label = classify(model, card)
            labels.append(label)

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
    # plt.subplot(121)
    ax.imshow(output)
    for contour, label in zip(contours, labels):
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        ax.text(
            cx, cy, label.replace("-", "\n"), fontsize="small", ha="center", va="center"
        )

    # output_cards = cv2.cvtColor(
    #     sk.util.montage(cards, channel_axis=-1, fill=[0, 0, 0], rescale_intensity=True),
    #     cv2.COLOR_BGR2RGB,
    # )
    # plt.subplot(122)
    # plt.imshow(output_cards)
    # plt.imshow(rectified_cards[1])
    # plt.show()


def camera_loop(args):
    rospy.init_node("vision", anonymous=True)
    bridge = cv_bridge.CvBridge()
    model = load_model(args.model, use_resnet=args.use_resnet)
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(10, 8))

    while True:
        exit = input(
            "Press ENTER to take camera image and detect cards, or anything to exit."
        )
        if exit:
            break

        image_data = rospy.wait_for_message("/usb_cam/image_raw", Image)
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")

        fig.clf()
        main_detect(image, model=model, fig=fig)


def main(args):
    if args.type == "image":
        image = cv2.imread(args.image)

        model = load_model(args.model, use_resnet=args.use_resnet)

        # detect once and exit
        main_detect(image, model=model)
        return
    elif args.type == "camera":
        # run loop listening to the camera
        camera_loop(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(dest="type")

    image_parser = subparser.add_parser("image")
    image_parser.add_argument("image", help="Input image")

    image_parser.add_argument(
        "--model",
        default=None,
        help="Model file to load for use as the card classifier",
    )
    image_parser.add_argument("--use-resnet", action="store_true")

    camera_parser = subparser.add_parser("camera")

    camera_parser.add_argument(
        "--model",
        default=None,
        help="Model file to load for use as the card classifier",
    )
    camera_parser.add_argument("--use-resnet", action="store_true")
    camera_parser.add_argument(
        "--live", action="store_true", help="Classify in real time"
    )

    main(parser.parse_args())
