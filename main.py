from datetime import datetime as dt

import torch
from torch import tensor, optim, topk, round
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, Normalize, Resize, \
    CenterCrop

from MLFnet import MLFnet
from dataset import MultimonDataset, CelebADataset


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

        train_dataset, \
        test_dataset, \
        valid_dataset = [CelebADataset(data_file=f".\\data\\{context}\\labels.txt", key_mask=partitions[type_],
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
                        {"type": "LazyLinear", "out_features": 512}, {"type": "ReLU"},
                        {"type": "Linear", "in_features": 512, "out_features": 512},
                        {"type": "Linear", "in_features": 512, "out_features": 1},
                        {"type": "Sigmoid"}] for task in tasks}

        losses = {task: BCELoss() for task in tasks}

        def get_accuracy(task, preds, labels):
            # all celeba tasks are binary so we use the same test for all
            return sum([p == l for p, l in zip(round(preds).tolist(), labels.tolist())])

    elif context == "multimon":
        with open("data\\multimon\\type_weights.txt", "r") as f:
            type_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            type_indexes = {line.split(":")[0]: index for index, line in enumerate(type_counts)}

        with open("data\\multimon\\gen_weights.txt", "r") as f:
            gen_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            gen_indexes = {line.split(":")[0]: index for index, line in enumerate(gen_counts)}

        train_dataset, \
        test_dataset, \
        valid_dataset = [MultimonDataset(data_file=f".\\data\\{context}\\labels.txt", key_mask=partitions[type_],
                                         type_dict=type_indexes, gen_dict=gen_indexes,
                                         img_path=f".\\data\\{context}\\data", device=device, no_mask=False,
                                         transforms=transforms[type_])
                         for type_ in ["train", "test", "valid"]]

        heads = {"Type": [{"type": "Flatten"},
                          {"type": "LazyLinear", "out_features": 512}, {"type": "ReLU"},
                          {"type": "Linear", "in_features": 512, "out_features": len(type_counts)}],
                 "Gen": [{"type": "Flatten"},
                         {"type": "LazyLinear", "out_features": 512}, {"type": "ReLU"},
                         {"type": "Linear", "in_features": 512, "out_features": len(gen_counts)}],
                 "Shiny": [{"type": "Flatten"},
                           {"type": "LazyLinear", "out_features": 512}, {"type": "ReLU"},
                           {"type": "Linear", "in_features": 512, "out_features": 1}, {"type": "Sigmoid"}]}
        losses = {"Type": BCEWithLogitsLoss(pos_weight=tensor([type_counts[t] / sum(type_counts.values())
                                                               for t in type_counts])).to(device),
                  "Gen": BCEWithLogitsLoss(pos_weight=tensor([gen_counts[g] / sum(gen_counts.values())
                                                              for g in gen_counts])).to(device),
                  "Shiny": BCELoss()}

        def get_accuracy(task, preds, labels):
            if task == "Type":
                return sum([sum(x in p_ind for x in l_ind) > 0 for p_ind, l_ind in
                            zip(topk(preds, dim=1, k=2).indices.tolist(),
                                topk(labels, dim=1, k=2).indices.tolist())])
            elif task == "Gen":
                return sum([p_ind == l_ind for p_ind, l_ind in
                            zip(topk(preds, dim=1, k=1).indices.tolist(),
                                topk(labels, dim=1, k=1).indices.tolist())])
            else:
                return sum([p == l for p, l in zip(round(preds).tolist(), labels.tolist())])
    else:
        raise ValueError("Invalid context")

    train_dl = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
    test_dl = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=True)
    valid_dl = DataLoader(batch_size=batch_size, dataset=valid_dataset, shuffle=True)

    return train_dl, test_dl, valid_dl, heads, losses, get_accuracy


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(f"{device=}")

    batch_size = 16
    context = "celeba"
    input_size = 64
    transforms = {
        'train': Compose([RandomResizedCrop(input_size), RandomHorizontalFlip(),
                          ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': Compose([Resize(input_size), CenterCrop(input_size),
                         ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': Compose([Resize(input_size), CenterCrop(input_size),
                          ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    train_dl, test_dl, valid_dl, \
    heads, losses, get_acc = get_context_parts(context, device, batch_size, input_size, transforms)

    layers = [{"type": "Conv2d", "in_channels": 3, "out_channels": 6, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 6, "out_channels": 12, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 12, "out_channels": 24, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 24, "out_channels": 48, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 48, "out_channels": 48, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 48, "out_channels": 48, "kernel_size": (3, 3),
               "stride": (1, 1), "padding": 0, "padding_mode": "zeros"}]

    model = MLFnet(tasks=tuple(heads.keys()), heads=heads, device=device)
    model.add_layer(None, **layers.pop(0))

    optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.25, verbose=True)

    epochs = len(layers) * 3

    out_para = ("Epoch {epoch}/{epochs}\n" +
                "\n".join([f"{phase.title():7} -- \t" +
                           " | ".join([f"{task.title()}:{{{phase}-{task}:.4%}}"
                                       for task in model.tasks])
                           for phase in ["train", "test"]]))

    out_file = f"stats\\data-{context}-{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    with open(out_file, "w") as f:
        f.write("epoch,batch,task,task_id,param_id,param_value\n")

    for epoch in range(epochs):
        stats = {f"{phase}-{task}": list([0, 0]) for task in model.tasks for phase in ["train", "test"]}
        if epoch % 2 == 0 and epoch > 0:
            model.freeze_model()
            new_layers = model.add_layer(None, **layers.pop(0))
            for layer in new_layers:
                optimiser.add_param_group({"params": layer.parameters()})
            scheduler.step()

        # using an internal loop for training/testing we avoid duplicating code
        for phase in ["train", "test", "validate"]:
            if phase == "train":
                data_loader = train_dl
            elif phase == "test":
                data_loader = test_dl
            else:
                data_loader = valid_dl

            for b_id, (data, labels) in enumerate(data_loader):
                data = data.to(device)
                # labels are all tensors which can be moved but the list itself cannot
                for task in model.tasks:
                    labels[task] = labels[task].to(device)

                optimiser.zero_grad()

                if phase == "train":  # only if we are training do we back prop
                    preds = model(data)

                    total_loss = 0.0
                    for task in model.tasks:
                        total_loss += losses[task](preds[task], labels[task])
                    total_loss.backward()

                    optimiser.step()

                elif phase == "test":  # otherwise just get results without gradients or back prop
                    with torch.no_grad():
                        preds = model(data)

                else:  # if validating also check grouping of tasks (and do branches etc)
                    preds = model(data)
                    ls = {head: losses[head](preds[head], labels[head]) for head in model.tasks}
                    grads = model.collect_param_grads(losses=ls)

                    with open(out_file, "a") as f:
                        for t_id, task in enumerate(model.tasks):
                            for g_id, grad in enumerate(grads[task]):
                                f.write(f"{epoch},{b_id},{task},{t_id},{g_id},{grad}\n")

                # ACCURACY
                if phase in ["train", "test"]:
                    for task in model.tasks:
                        acc = get_acc(task=task, preds=preds[task], labels=labels[task])
                        batch_size = preds[task].size()[0]
                        stats[phase + "-" + task][0] += acc
                        stats[phase + "-" + task][1] += batch_size

        for task in stats:
            stats[task] = stats[task][0] / stats[task][1]

        # Print results per epoch
        print(out_para.format(epoch=epoch + 1, epochs=epochs, **stats))


if __name__ == "__main__":
    main()
