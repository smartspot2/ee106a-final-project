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
from util.classify import classify_consensus
from util.detect import detect_cards, find_contours, find_card_center
from util.nn import CardClassifier, CardClassifierResnet
from util.set import find_set
from util.labels import label_from_string, deserialize_label

import image_geometry
from sensor_msgs.msg import Image, CameraInfo
from set_msgs.msg import Card
from set_msgs.srv import CardData
import cv_bridge


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
        for card_variations in cards:
            label, _ = classify_consensus(model, card_variations)
            labels.append(label)
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

    fig = plt.gcf()
    spec = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(2, 1))
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])

    ax1.imshow(output)
    for contour, label in zip(contours, labels):
        if label is None:
            # unable to classify, skip
            continue

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


def main_manual(model_file, use_resnet=False):
    """
    Main function with manual image capturing.
    """
    rospy.init_node("vision", anonymous=True)
    bridge = cv_bridge.CvBridge()
    model = load_model(model_file, use_resnet=use_resnet)

    while True:
        exit = input(
            "Press ENTER to take camera image and detect cards, or anything to exit."
        )
        if exit:
            break

        image_data = rospy.wait_for_message("/usb_cam/image_raw", Image)
        camera_info = rospy.wait_for_message("/usb_cam/camera_info", CameraInfo)
        camera_model = image_geometry.PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info)

        raw_image = bridge.imgmsg_to_cv2(image_data, "bgr8")
        image = raw_image.copy()
        camera_model.rectifyImage(raw_image, image)

        plt.clf()
        main_detect(image, model=model)
        plt.show()


def vision_callback(_request, model, tag_number):
    # get image from camera
    image_data = rospy.wait_for_message("/usb_cam/image_raw", Image)
    camera_info = rospy.wait_for_message("/usb_cam/camera_info", CameraInfo)
    camera_model = image_geometry.PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    bridge = cv_bridge.CvBridge()
    raw_image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    image = raw_image.copy()
    camera_model.rectifyImage(raw_image, image)

    # find contours
    contours = find_contours(image)

    # find cards
    cards = detect_cards(image, contours, variations=3)

    # classify cards
    labels = []
    for card_variations in cards:
        label, _ = classify_consensus(model, card_variations)
        labels.append(label)

    filtered_cards = []
    filtered_contours = []
    filtered_labels = []
    for card, contour, label in zip(cards, contours, labels):
        if label is None:
            continue

        filtered_cards.append(card)
        filtered_contours.append(contour)
        filtered_labels.append(label)

    # find the set
    found_set = find_set(filtered_labels)

    # prepare return message
    return_cards = []
    for idx, (card, contour, label) in enumerate(
        zip(filtered_cards, filtered_contours, filtered_labels)
    ):
        # find center of contour in AR tag coordinates
        card_center = find_card_center(contour, tag_number)
        card_center = card_center.flatten()

        # convert label to ints
        shape, color, count, shade = deserialize_label(label_from_string(label))

        cur_card = Card(
            shape=shape,
            color=color,
            number=count,
            shading=shade,
        )
        cur_card.position.x = card_center[0]
        cur_card.position.y = card_center[1]
        cur_card.position.z = card_center[2]
        return_cards.append(cur_card)

    return {"cards": return_cards, "set": found_set}


def main(tag_number, model_file, use_resnet=False):
    """
    Main function to start a service listening to service calls.
    """
    model = load_model(model_file, use_resnet=use_resnet)
    rospy.loginfo(f"Model file: {model_file}")
    assert model is not None

    rospy.Service(
        "/vision",
        CardData,
        partial(vision_callback, model=model, tag_number=tag_number),
    )
    rospy.loginfo("Running vision server...")
    rospy.spin()


MAIN_MANUAL = False

if __name__ == "__main__":
    rospy.init_node("vision", anonymous=True)

    if MAIN_MANUAL:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model-file")
        parser.add_argument("--use-resnet", action="store_true")
        args = parser.parse_args()

        main_manual(args.model_file, args.use_resnet)
    else:
        model_file = rospy.get_param("/vision/model_file")
        use_resnet = rospy.get_param("/vision/use_resnet", False)
        tag_number = rospy.get_param("/vision/tag_number")

        main(tag_number, model_file, use_resnet)
