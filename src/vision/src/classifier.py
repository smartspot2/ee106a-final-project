#!/usr/bin/env python

"""
Main class for training/running the classifier NN.
"""

import numpy as np
import skimage as sk
import torch
import tqdm
import util
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    InterpolationMode,
    RandomOrder,
    RandomResizedCrop,
    RandomRotation,
    ToDtype,
    ToImage,
)
from util import torch_util


def train(args):
    model = util.nn.CardClassifier().to(torch_util.device)
    if args.start_model is not None:
        model.load_state_dict(torch.load(args.start_model))

    data_transform = Compose(
        [
            ToImage(),
            ToDtype(torch.float, scale=True),
            RandomOrder(
                [
                    RandomResizedCrop(size=(280, 180), scale=(0.75, 1), antialias=True),
                    RandomRotation(
                        degrees=(-20, 20),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=1,
                    ),
                ]
            ),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]
    )
    dataset = util.nn.CardData(args.image_dir, transform=data_transform)
    eval_transform = Compose([ToImage(), ToDtype(torch.float, scale=True)])
    eval_dataset = util.nn.CardData(args.image_dir, transform=eval_transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm.trange(args.num_epochs):
        losses = []
        model.train()
        for images, labels in dataloader:
            # move to designated device
            images = images.to(torch_util.device)
            labels = labels.to(torch_util.device)

            output = model(images)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = np.mean(losses)
        if args.wandb:
            util.log.log_scalar(epoch, "train_loss", mean_loss)
        else:
            print("loss:", mean_loss)

        # evaluate on all normal images
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for images, labels in eval_dataloader:
                # move to designated device
                images = images.to(torch_util.device)
                labels = labels.to(torch_util.device)

                output = model(images)
                pred = output.argmax(1)
                total_correct += (pred == labels).sum().item()

            accuracy = total_correct / len(eval_dataset)

        if args.wandb:
            util.log.log_scalar(epoch, "eval_accuracy", accuracy)
        else:
            print("accuracy:", accuracy)

        torch.save(model.state_dict(), args.model_file)


def run(args):
    """
    Run a saved model on a given input image.
    """
    # from torchinfo import summary

    model = util.nn.CardClassifier().to(torch_util.device)
    model.load_state_dict(torch.load(args.model_file))
    # summary(model, input_size=(1, 3, 280, 180))

    raw_image = sk.util.img_as_float(sk.io.imread(args.image)).astype(np.float32)
    image = torch.tensor(raw_image).to(torch_util.device)

    # torch expects (channels, height, width)
    input_image = image.moveaxis((0, 1, 2), (1, 2, 0))[None, ...]

    pred_logits = model(input_image)
    pred = torch.argmax(pred_logits).item()

    string_label = util.nn.label_to_string(*util.nn.deserialize_label(pred))
    print(string_label)


def main(args):
    torch_util.init_device(use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    if args.wandb:
        util.log.init_wandb(args)

    if args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", default=False, action="store_true")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--wandb", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "image_dir", help="Image directory for labeled training images"
    )
    train_parser.add_argument("model_file", help="Trained model file to save to")
    train_parser.add_argument(
        "--start-model", default=None, help="Starting model state"
    )

    train_hyperparams = train_parser.add_argument_group("Hyperparameters")

    train_hyperparams.add_argument(
        "--num-epochs", default=100, type=int, help="Number of epochs to train for"
    )
    train_hyperparams.add_argument(
        "--batch-size", default=64, type=int, help="Training batch size"
    )
    train_hyperparams.add_argument(
        "--learning-rate", default=1e-4, type=float, help="Learning rate"
    )

    train_hyperparams.add_argument(
        "--eval-batch-size", default=128, type=int, help="Evaluation batch size"
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model_file", help="Saved model file")
    run_parser.add_argument("image", help="Image to classify")

    main(parser.parse_args())
