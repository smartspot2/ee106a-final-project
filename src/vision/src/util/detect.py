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


def grayworld_assumption(im):
    wb_im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    avg_a = np.median(wb_im[:, :, 1])
    avg_b = np.median(wb_im[:, :, 2])
    # print(avg_a)
    # print(avg_b)
    # print(wb_im.shape)
    a_delta = (avg_a - 128) * (wb_im[:, :, 0] / 255) * 1.1
    b_delta = (avg_b - 128) * (wb_im[:, :, 1] / 255) * 1.1
    wb_im[:, :, 1] = wb_im[:, :, 1] - a_delta
    wb_im[:, :, 2] = wb_im[:, :, 2] - b_delta
    return cv2.cvtColor(wb_im, cv2.COLOR_LAB2BGR)


def yen_thresholded_wb(im):
    wb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(wb_im)
    yen = sk.filters.threshold_yen(wb_im)
    print(yen)
    result = sk.exposure.rescale_intensity(im, (0, yen + 50), (0, 255)).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def reduce_bitdepth(im, bins):
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = sk.exposure.rescale_intensity(sk.util.img_as_float(image), out_range=(0, 1))

    # run k-means clustering to reduce bit-depth
    kmeans = KMeans(bins, n_init="auto")
    # flatten image and fit kmeans
    reduced_idx = kmeans.fit_predict(image.reshape(-1, 3))
    # use the closest cluster as the new color
    reduced = kmeans.cluster_centers_[reduced_idx]

    # use the most common as the background
    most_common_idx = scipy.stats.mode(reduced_idx)[0]
    background_mask = reduced_idx == most_common_idx
    reduced[background_mask] = 1

    reduced = reduced.reshape(image.shape)

    result = sk.util.img_as_ubyte(reduced).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def detect_cards(image, contours, card_shape=(180, 280)):
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
        # dst = grayworld_assumption(dst)
        # dst = yen_thresholded_wb(dst)
        dst = cv2.GaussianBlur(dst, (7, 7), 1)
        dst = reduce_bitdepth(dst, 3)
        # dst = segment(dst, 50)
        rectified_cards.append(dst)

    return rectified_cards
