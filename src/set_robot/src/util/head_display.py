import rospy
import cv_bridge
from sensor_msgs.msg import Image
import cv2
import skimage as sk
import numpy as np

WIDTH = 1024
HEIGHT = 600

def display_image(image):
    """
    Display an image on the sawyer.
    Image must be a cv2 image.
    """
    height, width, _ = image.shape
    width_pad = (WIDTH - width) // 2
    height_pad = (HEIGHT - height) // 2

    image = np.pad(image, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)))

    publisher = rospy.Publisher("/robot/head_display", Image, queue_size=1)
    bridge = cv_bridge.CvBridge()
    img_msg = bridge.cv2_to_imgmsg(image)

    for _ in range(10):
        publisher.publish(img_msg)
        rospy.sleep(0.1)

def display_file(file):
    image = cv2.imread(file)
    display_image(image)

if __name__ == "__main__":
    rospy.init_node("test")

    img = cv2.imread("labeled_images/diamond-green-1-outline.jpg")
    print(img.shape)
    display_image(img)