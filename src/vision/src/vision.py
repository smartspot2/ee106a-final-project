"""
Vision node to detect Set cards.
"""

import cv2
import cv2.typing
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk


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


def detect_cards(image):
    contours = find_contours(image)

    output = image.copy()
    for idx in range(len(contours)):
        rgb = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        output = cv2.drawContours(output, contours, idx, rgb, 2, cv2.LINE_8)

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
                [0, 300],
                [180, 300],
                [180, 0],
            ]
        ).astype(np.float32)

        M = cv2.getPerspectiveTransform(oriented_contour, rectified_target)
        dst = cv2.warpPerspective(image, M, (180, 300))

        dst = sk.exposure.adjust_gamma(dst)
        rectified_cards.append(dst)


    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(output)

    output_cards = cv2.cvtColor(sk.util.montage(rectified_cards, channel_axis=-1, fill=[0, 0, 0], rescale_intensity=True), cv2.COLOR_BGR2RGB)
    plt.subplot(122)
    plt.imshow(output_cards)
    plt.show()


def main(args):
    image = cv2.imread(args.image)

    detect_cards(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Input image")

    main(parser.parse_args())
