from MLFnet import MLFnet


def main():
    model = MLFnet(tasks=("a", "b", "c"), heads=None)
    print(model)
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 1, "out_channels": 256, "kernel_size": (3, 3)})
    print(model)
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 1, "out_channels": 256, "kernel_size": (3, 3)})
    model.freeze_model()
    model.add_layer(target_group=None,
                    **{"type": "Conv2d", "in_channels": 1, "out_channels": 256, "kernel_size": (3, 3)})
    print(model)
    print(model.frozen_states())


if __name__ == "__main__":
    main()
