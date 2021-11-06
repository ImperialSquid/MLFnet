import torch
from torch import nn

from MLFnet import MLFnet


def main():
    model = MLFnet(tasks=("a", "b", "c"), heads=None)


if __name__ == "__main__":
    main()