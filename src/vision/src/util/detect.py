import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import skimage as sk
from sklearn.cluster import MiniBatchKMeans
import tf2_ros
import rospy
import tf

from sensor_msgs.msg import CameraInfo

ARCLENGTH_MAX_EPS = 1
CLOSE_COLORS_THRESH = 0.1
SIDE_LENGTH_EPS = -0.05


def simplify_contour(contour, n_corners=4):
    """
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    Taken from https://stackoverflow.com/a/55339684

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns None.
    """
    n_iter, max_iter = 0, 100
    lb, ub = 0.0, ARCLENGTH_MAX_EPS

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return None

        k = (lb + ub) / 2.0
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.0
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.0
        else:
            return approx


def find_contours(image):
    # grayscale image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # normalize image
    image_gray = cv2.normalize(image_gray, image_gray, 0, 255, cv2.NORM_MINMAX)

    # canny edge detection
    edges = cv2.Canny(image_gray, 50, 200, L2gradient=True)

    # find contours using the edge image
    contours, _ = cv2.findContours(
        edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1
    )

    # output = image.copy()
    final_contours = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 4000:
            # rgb = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            # output = cv2.drawContours(output, contours, idx, rgb, 2, cv2.LINE_8)

            # approximate the contour with a simpler polygon
            approx_contour = simplify_contour(contour)
            if approx_contour is None:
                # unable to simplify to 4 corners
                continue

            approx_contour_area = cv2.contourArea(approx_contour)
            if approx_contour_area > 4000 and len(approx_contour) == 4:
                # approximate contour area must still be large enough,
                # and it should be a quadrilateral
                final_contours.append(approx_contour)

    # sk.io.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # sk.io.show()
    return final_contours


def debug_plot_image(image, centers, labels, bins):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for bin in range(bins):
        vals = image.reshape(-1, 3)[labels == bin]
        ax.scatter(
            vals[:, 0],
            vals[:, 1],
            vals[:, 2],
            marker="+",
            color=tuple(centers[bin].flatten()),
            alpha=0.2,
        )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c="black",
    )
    plt.show()


def reduce_bitdepth(im, bins):
    """
    Reduce the bitdepth of an image using k-means clustering.

    This function is non-deterministic, since the k-means clustering uses a random initialization.
    """
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = sk.exposure.rescale_intensity(sk.util.img_as_float(image), out_range=(0, 1))

    # run k-means clustering to reduce bit-depth
    kmeans = MiniBatchKMeans(bins, n_init="auto")
    # flatten image and fit kmeans
    flat_image = image.reshape(-1, 3)
    reduced_idx = kmeans.fit_predict(flat_image)
    centers = kmeans.cluster_centers_

    # use the closest cluster as the new color
    reduced = centers[reduced_idx]

    # use the most common as the background
    values, counts = np.unique(reduced_idx, return_counts=True)
    counts_argsorted = np.argsort(counts)
    most_common_idx = values[counts_argsorted[-1]]
    second_common_idx = values[counts_argsorted[-2]]

    background_mask = reduced_idx == most_common_idx
    reduced[background_mask] = (1, 1, 1)

    # if the second most common is very close to the most common, also make it white
    if (
        np.linalg.norm(centers[most_common_idx] - centers[second_common_idx])
        < CLOSE_COLORS_THRESH
    ):
        background_mask = reduced_idx == second_common_idx
        reduced[background_mask] = (1, 1, 1)

    reduced = reduced.reshape(image.shape)

    # debug_plot_image(image_lab, centers, reduced_idx, bins)

    result = sk.util.img_as_ubyte(
        sk.exposure.rescale_intensity(np.clip(reduced, 0, 1), out_range=(0, 1))
    ).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def inflate_y(pos):
    """
    Inflate the y-value a little bit if it is smaller, to counteract the perspective distortion slightly.
    """
    y = pos[1]
    new_y = y * (1 + 5 / (0.1 * y + 1))
    return np.array([pos[0], new_y])


def find_card_center(contour, tag_number):
    """
    Returns the center of a given contour in AR tag coordinates, as well as the u and v position in the image.
    """
    M = cv2.moments(contour)

    # the point in the image coordinates
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])

    camera_info = rospy.wait_for_message("/usb_cam/camera_info", CameraInfo)
    K = camera_info.K
    f = K[0]

    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)

    try:
        # TODO: lookup the transform and save it in trans
        trans = tfBuffer.lookup_transform(
            "usb_cam",
            "ar_marker_{}".format(tag_number),
            rospy.Time(),
            rospy.Duration(10),
        )
    except Exception as e:
        raise e

    # transformation matrix from AR marker, as well as its inverse
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ]
    )

    x, y, z = (
        trans.transform.translation.x,
        trans.transform.translation.y,
        trans.transform.translation.z,
    )

    cam2ar = np.array(
        [
            [
                np.cos(roll) * np.cos(pitch),
                np.sin(roll) * -1 * np.cos(pitch),
                np.sin(pitch),
                x,
            ],
            [
                np.cos(yaw) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw),
                np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(pitch) * np.sin(yaw),
                -1 * np.cos(pitch) * np.sin(yaw),
                y,
            ],
            [
                -1 * np.cos(roll) * np.cos(yaw) * np.sin(pitch)
                + np.sin(roll) * np.sin(yaw),
                np.cos(yaw) * np.sin(roll) * np.sin(pitch) + np.cos(roll) * np.sin(yaw),
                np.cos(roll) * np.cos(yaw),
                z,
            ],
            [0, 0, 0, 1],
        ]
    )

    ar2cam = np.linalg.inv(cam2ar)

    # normal to plane, point on the plane
    normal = ar2cam @ np.array([[0, 0, 1, 0]]).T
    p0 = ar2cam @ np.array([0, 0, 0, 1]).T

    # origin and direction of ray
    o = np.array([[0, 0, 0, 1]]).T
    d = np.array([u, v, f, 0]).T

    # point of intersection in camera space
    p = o + d * (((p0 - o) @ normal) / (d @ normal))

    return cam2ar @ p


def detect_cards(image, contours, card_shape=(180, 280), variations=1, preprocess=True):
    """
    Detect all cards in an image, given the card contours.

    `variations` denotes the number of preprocessing variations to perform.
        The added variations can help with robustness in the classification algorithm.

    `preprocess` is a flag for whether this function should preprocess the image through
        color bitdepth reduction; if false, the raw rectified cards are returned,
        and `variations` is ignored.
    """
    card_width, card_height = card_shape
    # rectify images
    rectified_cards = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        # orient the contour to be counter-clockwise, starting from top left point
        # this should be the default when detecting contours, but we want to make sure
        min_idx = np.argmin(np.linalg.norm(contour, axis=1))

        next_idx = (min_idx + 1) % 4
        prev_idx = (min_idx - 1) % 4

        prev_vec = inflate_y(contour[prev_idx]) - inflate_y(contour[min_idx])
        next_vec = inflate_y(contour[next_idx]) - inflate_y(contour[min_idx])
        cross_prod = np.cross(prev_vec, next_vec)

        # positive cross product => counter-clockwise
        if cross_prod < 0:
            # flip if negative
            next_idx, prev_idx = prev_idx, next_idx

        # check the lengths of each side; if prev side is longer than next side, then shift by one
        if np.linalg.norm(prev_vec) > np.linalg.norm(next_vec):
            min_idx = (min_idx + 1) % 4
            next_idx = (next_idx + 1) % 4
            prev_idx = (prev_idx + 1) % 4

        oriented_contour = np.array(
            [
                contour[min_idx],
                contour[next_idx],
                contour[(next_idx + 1) % 4],
                contour[prev_idx],
            ],
        ).astype(np.float32)

        rectified_target = np.array(
            [
                [0, 0],
                [0, card_height],
                [card_width, card_height],
                [card_width, 0],
            ]
        ).astype(np.float32)

        M = cv2.getPerspectiveTransform(oriented_contour, rectified_target)
        dst = cv2.warpPerspective(image, M, card_shape)
        if preprocess:
            dst = cv2.GaussianBlur(dst, (7, 7), 1)
            if variations == 1:
                # if only one variation, then reduce once and append
                dst = reduce_bitdepth(dst, 3)
                rectified_cards.append(dst)
            else:
                # if multiple variations, reduce bitdepth multiple times to add variations
                dst_variations = [reduce_bitdepth(dst, 3) for _ in range(variations)]
                rectified_cards.append(dst_variations)
        else:
            # don't preprocess cards; just return the raw warped image
            rectified_cards.append(dst)

    return rectified_cards
