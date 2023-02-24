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


def get_context_parts(context, device, batch_size, transforms):
    partitions = {"train": [], "test": [], "valid": []}
    with open(f"./data/{context}/partitions.txt") as f:
        for line in f.readlines():
            if line.split(",")[1].strip() == "0":
                partitions["train"].append(line.split(",")[0])
            elif line.split(",")[1].strip() == "1":
                partitions["valid"].append(line.split(",")[0])
            elif line.split(",")[1].strip() == "2":
                partitions["test"].append(line.split(",")[0])

    if context == "celeba":
        load_fraction = 0.1

        train_dataset, test_dataset, valid_dataset = \
            [CelebADataset(data_file=f".\\data\\{context}\\labels.txt", key_mask=partitions[type_],
                           img_path=f".\\data\\{context}\\data", device=device, no_mask=False,
                           transforms=transforms[type_], load_fraction=load_fraction)
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

    elif context == "multimon":
        train_dataset, test_dataset, valid_dataset = \
            [MultimonDataset(data_file="data.csv", part_file="partitions.csv", img_path=".\\data\\multimon\\sprites",
                             device=device, transforms=transforms[part], output_size=64, partition=part)
             for part in ["train", "test", "valid"]]

        gen_counts = train_dataset.gen_counts
        type_counts = train_dataset.type_counts

        gen_indexes = train_dataset.gen_indexes
        type_indexes = train_dataset.type_indexes

        d1 = {"Type": [{"type": "Flatten"},
                       {"type": "LazyLinear", "out_features": len(type_counts)}],
              "Gen": [{"type": "Flatten"},
                      {"type": "LazyLinear", "out_features": len(gen_counts)}]}
        d2 = {task.title(): [{"type": "Flatten"},
                             {"type": "LazyLinear", "out_features": 1}]
              for task in ["hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"]}
        heads = {**d1, **d2}

        d1 = {"Type": BCEWithLogitsLoss(pos_weight=tensor([type_counts[t] / sum(type_counts.values())
                                                           for t in type_counts])).to(device),
              "Gen": BCEWithLogitsLoss(pos_weight=tensor([gen_counts[g] / sum(gen_counts.values())
                                                          for g in gen_counts])).to(device)}
        d2 = {task.title(): MSELoss().to(device)
              for task in ["hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"]}
        losses = {**d1, **d2}

        def get_accuracy(task, preds, labels):
            if task == "Type":  # sum of predictions where at least one is correct
                return sum([sum(x in p_ind for x in l_ind) > 0 for p_ind, l_ind in
                            zip(topk(preds, dim=1, k=2).indices.tolist(),
                                topk(labels, dim=1, k=2).indices.tolist())])
            elif task == "Gen":  # sum of predictions where gen is correct
                return sum([p_ind == l_ind for p_ind, l_ind in
                            zip(topk(preds, dim=1, k=1).indices.tolist(),
                                topk(labels, dim=1, k=1).indices.tolist())])
            else:  # MSE since accuracy is not a simple metric for regression tasks
                return mean([(p - l) ** 2 for p, l in zip(round(preds).tolist(), labels.tolist())])
    else:
        raise ValueError("Invalid context")

    train_dl = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
    test_dl = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=True)
    valid_dl = DataLoader(batch_size=batch_size, dataset=valid_dataset, shuffle=True)

    return train_dl, test_dl, valid_dl, heads, losses, get_accuracy
