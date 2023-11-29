import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import skimage as sk
from sklearn.cluster import KMeans


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

    final_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 4000:
            # approximate the contour with a simpler polygon
            arclength = cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, 0.05 * arclength, True)
            approx_contour_area = cv2.contourArea(approx_contour)

            if approx_contour_area > 2000 and len(approx_contour) == 4:
                # approximate contour area must still be large enough,
                # and it should be a quadrilateral
                final_contours.append(approx_contour)

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
    kmeans = KMeans(bins, n_init="auto")
    # flatten image and fit kmeans
    flat_image = image.reshape(-1, 3)
    reduced_idx = kmeans.fit_predict(flat_image)
    centers = kmeans.cluster_centers_

    # use the closest cluster as the new color
    reduced = centers[reduced_idx]

    # use the most common as the background
    most_common_idx = scipy.stats.mode(reduced_idx, keepdims=True)[0]
    background_mask = reduced_idx == most_common_idx
    reduced[background_mask] = (1, 1, 1)

    reduced = reduced.reshape(image.shape)

    # debug_plot_image(image_lab, centers, reduced_idx, bins)

    result = sk.util.img_as_ubyte(np.clip(reduced, 0, 1)).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def detect_cards(image, contours, card_shape=(180, 280), variations=1):
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

        prev_vec = contour[prev_idx] - contour[min_idx]
        next_vec = contour[next_idx] - contour[min_idx]
        cross_prod = np.cross(prev_vec, next_vec)

        # positive cross product => counter-clockwise
        if cross_prod < 0:
            # flip if negative
            next_idx, prev_idx = prev_idx, next_idx

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
        dst = cv2.GaussianBlur(dst, (7, 7), 1)
        if variations == 1:
            # if only one variation, then reduce once and append
            dst = reduce_bitdepth(dst, 3)
            rectified_cards.append(dst)
        else:
            # if multiple variations, reduce bitdepth multiple times to add variations
            dst_variations = [reduce_bitdepth(dst, 3) for _ in range(variations)]
            rectified_cards.append(dst_variations)

    return rectified_cards
