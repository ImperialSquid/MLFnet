import torch

from MLFnet import MLFnet


def main():
    model = MLFnet(tasks=("a", "b", "c"), heads=None)
    print(model)
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 3, "out_channels": 256, "kernel_size": (3, 3)})
    print(model)
    # model(torch.zeros(1, 3, 96, 96))
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3)})
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3)})
    print(model)
    # model(torch.zeros(1, 3, 96, 96))
    model.split_group(old_group=("a", "b", "c"), new_groups=[("a", "b"), ("c",)])
    print(model)
    # model(torch.zeros(1, 3, 96, 96))
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3)})
    print(model)
    # model(torch.zeros(1, 3, 96, 96))

    # print(model.frozen_states())

    model.draw(torch.zeros(1, 3, 96, 96), verbose=True)


if __name__ == "__main__":
    main()
