import rospy
import cv_bridge
from sensor_msgs.msg import Image
import cv2
import skimage as sk
import numpy as np

WIDTH = 1024
HEIGHT = 600
PAD = 50

def display_image(image):
    """
    Display an image on the sawyer.
    Image must be a cv2 image.
    """
    height, width, _ = image.shape

    target_height = HEIGHT - 2 * PAD
    target_width = WIDTH - 2 * PAD
    height_diff = target_height - height
    width_diff = target_width - width

    if height_diff < width_diff:
        # resize to fit height
        ratio = target_height / height
        desired_height = target_height
        desired_width = width * ratio
    else:
        # resize to fit width
        ratio = target_width / width
        desired_height = height * ratio
        desired_width = target_width
    
    image = cv2.resize(image, (int(desired_width), int(desired_height)))

    height, width, _ = image.shape
    width_pad = int((WIDTH - width) // 2) + 1
    height_pad = int((HEIGHT - height) // 2) + 1

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

def clear_display():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    display_image(image)

def display_series(card_files, index=0):
    COLOR = (0, 255, 0)
    BORDER = 5

    cards = [cv2.imread(file) for file in card_files]
    current_card = cards[index]
    current_card[:BORDER, :] = COLOR
    current_card[-BORDER:, :] = COLOR
    current_card[:, :BORDER] = COLOR
    current_card[:, -BORDER:] = COLOR

    concatenated = np.concatenate(cards, axis=1)
    display_image(concatenated)

if __name__ == "__main__":
    rospy.init_node("test")

    # display_series([
    #     "labeled_images/diamond-green-1-solid.jpg",
    #     "labeled_images/diamond-red-2-solid.jpg",
    #     "labeled_images/diamond-purple-3-solid.jpg",
    # ], index=2)

    # clear_display()
    display_file("display_images/looking.png")