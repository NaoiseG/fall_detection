import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO


def visualize_bn_gamma_distribution(model, output_path):
    """
    Extract gamma weights from Batch Normalization layers of the model
    and visualize their distribution.

    Args:
        model: The loaded YOLO model.
        output_path: Path to save the histogram image.
    """
    gamma_weights = []

    for layer in model.modules():
        if hasattr(layer, "weight") and isinstance(layer, torch.nn.BatchNorm2d):
            gamma_weights.append(layer.weight.detach().cpu().numpy())

    gamma_weights = np.concatenate(gamma_weights)

    plt.figure(figsize=(8, 5))
    plt.hist(gamma_weights, bins=100, color="blue", alpha=0.7)
    plt.title("Distribution of Gamma Weights (BN Layers)")
    plt.xlabel("Gamma Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train-sparsity/weights/best.pt", help="model weights path")
    parser.add_argument("--output", type=str, default="bn-distribution.jpg", help="output histogram image path")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    visualize_bn_gamma_distribution(model, opt.output)


if __name__ == "__main__":
    main(parse_opt())
