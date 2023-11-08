#!/usr/bin/env python

"""
Small script to take snapshots from a USB camera for data collection.
"""
import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import os
from datetime import datetime


def main(args):
    rospy.init_node("take_snapshot", anonymous=True)

    bridge = cv_bridge.CvBridge()

    while True:
        input("Press ENTER to take a snapshot")

        # wait for a message
        image_data = rospy.wait_for_message("/usb_cam/image_raw", Image)

        # convert image message into cv2 format
        image = bridge.imgmsg_to_cv2(image_data, "bgr8")

        #cv2.imshow('image', image)
        # cv2.waitKey(0)

        # save image to a file
        cur_time = datetime.now()
        cur_time_formatted = cur_time.strftime(r"%Y%m%d_%H%M%S_%f")
        filename = f"{cur_time_formatted}.jpg"
        image_path = os.path.join(args.out_folder, filename)
        print("Image saved to", image_path)
        cv2.imwrite(image_path, image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("out_folder", help="Output folder for snapshot images")

    main(parser.parse_args())
