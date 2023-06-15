import torch
from torch import nn
from torch.hub import load
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from dataset import CelebADataset


class ModelMixin:
    def draw(self, input_tensor, filename=None, filetype="png", transforms="default", **options):
        import hiddenlayer as hl
        import os

        if filename is None:
            filename = self.__class__.__name__

        if options.get("verbose", False):
            transforms = []
            filename = filename + "_Verbose"

        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # make graphviz discoverable
        graph = hl.build_graph(self, input_tensor, transforms=transforms)
        for option in options:
            graph.theme[option] = options[option]
        graph.save(filename, format=filetype)


def get_context_parts(context, batch_size, transforms):
    if context == "celeba":
        train_dataset, test_dataset, valid_dataset = \
            [CelebADataset("./data/celeba", split=type_, transform=transforms[type_], target_transform=None)
             for type_ in ["train", "test", "valid"]]

        tasks = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                 "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                 "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                 "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                 "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
                 "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                 "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

        heads = {task: [{"type": "AdaptiveAvgPool2d", "output_size": (7, 7)},
                        {"type": "Linear", "in_features": 512 * 7 * 7, "out_features": 4096},
                        {"type": "ReLU", "inplace": True},
                        {"type": "Dropout", "p": 0.5},
                        {"type": "Linear", "in_features": 4096, "out_features": 4096},
                        {"type": "ReLU", "inplace": True},
                        {"type": "Dropout", "p": 0.5},
                        {"type": "Linear", "in_features": 4096, "out_features": 1}]
                 for task in tasks}

        losses = {task: BCELoss() for task in tasks}

        metrics = {phase: {task: MetricCollection([BinaryAccuracy(), BinaryF1Score()])
                           for task in tasks} for phase in ["train", "test"]}
    else:
        raise ValueError("Invalid context")

    train_dl = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
    test_dl = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=True)
    valid_dl = DataLoader(batch_size=batch_size, dataset=valid_dataset, shuffle=True)

    return train_dl, test_dl, valid_dl, heads, losses, metrics


def get_backbone_layers(model="vgg13", device=None):
    if device is None:
        raise ValueError("Device must be specified")

    backbone, blocks = None, None

    if model == "vgg13":
        model = load('pytorch/vision:v0.14.0', 'vgg13', weights="DEFAULT")
        backbone = nn.Sequential(model.features[0:5]).to(device)
        blocks = [nn.Sequential(model.features[5:10]).to(device),
                  nn.Sequential(model.features[10:15]).to(device),
                  nn.Sequential(model.features[15:20]).to(device),
                  nn.Sequential(model.features[20:25]).to(device)]
    elif model == "vgg19":
        model = load('pytorch/vision:v0.14.0', 'vgg19', weights="DEFAULT")
        backbone = nn.Sequential(model.features[0:5]).to(device)
        blocks = [nn.Sequential(model.features[5:10]).to(device),
                  nn.Sequential(model.features[10:18]).to(device),
                  nn.Sequential(model.features[18:28]).to(device),
                  nn.Sequential(model.features[28:36]).to(device)]

    return backbone, blocks


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    get_context_parts("celeba", device=device)
