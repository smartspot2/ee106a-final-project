import os

import cv2
import tqdm
from util.detect import reduce_bitdepth


def main(args):
    for file in tqdm.tqdm(os.listdir(args.input_dir)):
        input_path = os.path.join(args.input_dir, file)
        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(args.out_dir, file)

        image = cv2.imread(input_path)

        # reduce bitdepth
        reduced = reduce_bitdepth(image, bins=3)

        cv2.imwrite(output_path, reduced)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("out_dir", help="Output directory")

    main(parser.parse_args())
