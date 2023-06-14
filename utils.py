from statistics import mean

from torch import round, tensor, topk
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from dataset import CelebADataset, MultimonDataset


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
            [CelebADataset("./data/celeba", split=type_, transform=transforms, target_transform=None)
             for type_ in ["train", "test", "valid"]]

        tasks = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                 "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                 "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                 "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                 "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
                 "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                 "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"][:3]

        heads = {task: [{"type": "Flatten"},
                        {"type": "LazyLinear", "out_features": 1},
                        {"type": "Sigmoid"}] for task in tasks}

        losses = {task: BCELoss() for task in tasks}

        def get_accuracy(task, preds, labels):
            # all celeba tasks are binary so we use the same test for all
            return sum([p == l for p, l in zip(round(preds).tolist(), labels.tolist())])
    else:
        raise ValueError("Invalid context")

    train_dl = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
    test_dl = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=True)
    valid_dl = DataLoader(batch_size=batch_size, dataset=valid_dataset, shuffle=True)

    return train_dl, test_dl, valid_dl, heads, losses, get_accuracy
