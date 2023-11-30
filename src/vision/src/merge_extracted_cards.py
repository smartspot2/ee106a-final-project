import os
import shutil

from tqdm import tqdm

LABEL_FORMAT = "{shape}-{color}-{number}-{shade}"


def main(args):
    dest_files = set()
    for file in os.listdir(args.dest):
        filename, _ = os.path.splitext(file)
        dest_files.add(filename)

    for file in tqdm(os.listdir(args.source)):
        filename, ext = os.path.splitext(file)
        full_file = os.path.join(args.source, file)

        shape, color, number, shade = filename.split("-")

        if "_" in shade:
            shade, _ = shade.split("_")

        label = LABEL_FORMAT.format(
            shape=shape, color=color, number=number, shade=shade
        )

        nonce = None
        while True:
            if nonce is None:
                dest = os.path.join(args.dest, f"{label}{ext}")
            else:
                dest = os.path.join(args.dest, f"{label}_{nonce}{ext}")

            if os.path.isfile(dest):
                nonce = 1 if nonce is None else nonce + 1
            else:
                shutil.copyfile(full_file, dest)
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("dest")

    main(parser.parse_args())
