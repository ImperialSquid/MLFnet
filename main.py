import torch

from MLFnet import MLFnet


def main():
    model = MLFnet(tasks=("a", "b", "c"), heads=None)
    print(model.compiled_body[("a", "b")](torch.zeros(1, 3, 3, 3)))


if __name__ == "__main__":
    main()