import torch
from torch import tensor, optim, topk, round
from torch.nn import BCELoss, Sequential, Flatten, LazyLinear, Linear, Sigmoid, ReLU, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine

from MLFnet import MLFnet
from dataset import MultimonDataset, CelebADataset


def get_context_parts(context, device, batch_size):
    partitions = {"train": [], "test": [], "valid": []}
    with open(f"./data/{context}/partitions.txt") as f:
        for line in f.readlines():
            if line.split(",")[1].strip() == "0":
                partitions["train"].append(line.split(",")[0])
            elif line.split(",")[1].strip() == "1":
                partitions["valid"].append(line.split(",")[0])
            elif line.split(",")[1].strip() == "2":
                partitions["test"].append(line.split(",")[0])

    random_transforms_list = [RandomResizedCrop(size=96, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)),
                              RandomHorizontalFlip(1), RandomVerticalFlip(1),  # Force a flip if selected
                              RandomAffine(degrees=45),  # picks an angle between -45 and +45
                              RandomAffine(degrees=0, shear=[-30, 30, 0, 0]),  # X shear
                              RandomAffine(degrees=0, shear=[0, 0, -30, 30]),  # X shear
                              RandomAffine(degrees=0, translate=[0.2, 0.2])]  # X shear

    if context == "celeba":
        train_dataset = CelebADataset(data_file=f"./data/{context}/labels.txt", key_mask=partitions["train"],
                                      img_path=f"./data/{context}/data", device=device, no_mask=False,
                                      random_transforms=2, random_transforms_list=random_transforms_list)
        test_dataset = CelebADataset(data_file=f"./data/{context}/labels.txt", key_mask=partitions["test"],
                                     img_path=f"./data/{context}/data", device=device, no_mask=False,
                                     random_transforms=2, random_transforms_list=random_transforms_list)
        valid_dataset = CelebADataset(data_file=f"./data/{context}/labels.txt", key_mask=partitions["valid"],
                                      img_path=f"./data/{context}/data", device=device, no_mask=False,
                                      random_transforms=2, random_transforms_list=random_transforms_list)

        tasks = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                 "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                 "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                 "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                 "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
                 "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                 "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

        heads = {task: Sequential(Flatten(),
                                  LazyLinear(out_features=1024), ReLU(),
                                  Linear(in_features=1024, out_features=1024), ReLU(),
                                  Linear(in_features=1024, out_features=1), ReLU(),
                                  Sigmoid()).to(device) for task in tasks}

        losses = {task: BCELoss() for task in tasks}

    elif context == "multimon":
        with open("data\\multimon\\type_weights.txt", "r") as f:
            type_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            type_indexes = {line.split(":")[0]: index for index, line in enumerate(type_counts)}

        with open("data\\multimon\\gen_weights.txt", "r") as f:
            gen_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            gen_indexes = {line.split(":")[0]: index for index, line in enumerate(gen_counts)}

        train_dataset = MultimonDataset(data_file=f".\\data\\{context}\\labels.txt", device=device,
                                        type_dict=type_indexes, gen_dict=gen_indexes,
                                        key_mask=partitions["train"], img_path=f".\\data\\{context}\\data",
                                        random_transforms=2, random_transforms_list=random_transforms_list)
        test_dataset = MultimonDataset(data_file=f".\\data\\{context}\\labels.txt", device=device,
                                       type_dict=type_indexes, gen_dict=gen_indexes,
                                       key_mask=partitions["test"], img_path=f".\\data\\{context}\\data",
                                       random_transforms=2, random_transforms_list=random_transforms_list)
        valid_dataset = MultimonDataset(data_file=f".\\data\\{context}\\labels.txt", device=device,
                                        type_dict=type_indexes, gen_dict=gen_indexes,
                                        key_mask=partitions["valid"], img_path=f".\\data\\{context}\\data",
                                        random_transforms=2, random_transforms_list=random_transforms_list)

        heads = {"Type": Sequential(Flatten(),
                                    LazyLinear(out_features=1024), ReLU(),
                                    Linear(in_features=1024, out_features=1024), ReLU(),
                                    Linear(in_features=1024, out_features=len(type_counts))),
                 "Gen": Sequential(Flatten(),
                                   LazyLinear(out_features=1024), ReLU(),
                                   Linear(in_features=1024, out_features=1024), ReLU(),
                                   Linear(in_features=1024, out_features=len(gen_counts))),
                 "Shiny": Sequential(Flatten(),
                                     LazyLinear(out_features=1024), ReLU(),
                                     Linear(in_features=1024, out_features=1024), ReLU(),
                                     Linear(in_features=1024, out_features=1024), ReLU(),
                                     Linear(in_features=1024, out_features=1), Sigmoid())}
        losses = {"Type": BCEWithLogitsLoss(pos_weight=tensor([type_counts[t] / sum(type_counts.values())
                                                               for t in type_counts])).to(device),
                  "Gen": BCEWithLogitsLoss(pos_weight=tensor([gen_counts[g] / sum(gen_counts.values())
                                                              for g in gen_counts])).to(device),
                  "Shiny": BCELoss()}

    train_dataloader = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
    test_dataloader = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=True)
    valid_dataloader = DataLoader(batch_size=batch_size, dataset=valid_dataset, shuffle=True)

    return train_dataloader, test_dataloader, valid_dataloader, heads, losses


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(f"{device=}")

    batch_size = 32
    train_dataloader, test_dataloader, valid_dataloader, heads, losses = get_context_parts("multimon", device,
                                                                                           batch_size)

    model = MLFnet(tasks=tuple(heads.keys()), heads=heads, device=device)
    model.add_layer(None,
                    **{"type": "Conv2d", "in_channels": 3, "out_channels": 6, "kernel_size": (3, 3), "stride": (1, 1),
                       "padding": 0, "padding_mode": "zeros"})

    layers = [{"type": "Conv2d", "in_channels": 6, "out_channels": 12, "kernel_size": (3, 3), "stride": (1, 1),
               "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 12, "out_channels": 24, "kernel_size": (3, 3), "stride": (1, 1),
               "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 24, "out_channels": 48, "kernel_size": (3, 3), "stride": (1, 1),
               "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 48, "out_channels": 48, "kernel_size": (3, 3), "stride": (1, 1),
               "padding": 0, "padding_mode": "zeros"},
              {"type": "Conv2d", "in_channels": 48, "out_channels": 48, "kernel_size": (3, 3), "stride": (1, 1),
               "padding": 0, "padding_mode": "zeros"}]

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # TODO adaptive lr, big jumps when first starting

    epochs = len(layers) * 3
    stats_history = {"train-type-all": [], "train-type-any": [], "train-gen": [], "train-shiny": [],
                     "test-type-all": [], "test-type-any": [], "test-gen": [], "test-shiny": []}
    for epoch in range(epochs):
        stats = dict()
        if epoch % 3 == 0 and epoch > 0:
            model.freeze_model()
            model.add_layer(None, **layers[epoch // 3 - 1])

        # using an internal loop for training/testing we avoid duplicating code
        for phase in ["train", "test", "validate"]:
            if phase == "train":
                data_loader = train_dataloader
            elif phase == "test":
                data_loader = test_dataloader
            else:
                data_loader = valid_dataloader

            for data, labels in data_loader:
                data = data.to(device)
                # labels are all tensors which can be moved but the list itself cannot
                labels["Type"] = labels["Type"].to(device)
                labels["Gen"] = labels["Gen"].to(device)
                labels["Shiny"] = labels["Shiny"].to(device)

                optimizer.zero_grad()

                if phase == "train":  # only if we are training do we back prop
                    preds = model(data)

                    type_loss = losses["Type"](preds["Type"], labels["Type"])
                    gen_loss = losses["Gen"](preds["Gen"], labels["Gen"])
                    shiny_loss = losses["Shiny"](preds["Shiny"], labels["Shiny"])
                    total_loss = type_loss + gen_loss + shiny_loss
                    total_loss.backward()

                    optimizer.step()
                elif phase == "test":  # otherwise just get results without gradients or back prop
                    with torch.no_grad():
                        preds = model(data)
                else:  # if validating also check grouping of tasks (and do branches etc)
                    preds = model(data)
                    ls = {head: losses[head](preds[head], labels[head]) for head in ["Type", "Gen", "Shiny"]}
                    model.assess_grouping(method="agglomerative_clustering", losses=ls, debug=True)

                # ACCURACY
                # TODO try more advanced metrics f1 etc?
                # For All Type accuracy, check if the top 2 indices are *identical*
                type_cor_all = sum([p_ind == l_ind for
                                    p_ind, l_ind in zip(topk(preds["Type"], dim=1, k=2).indices.tolist(),
                                                        topk(labels["Type"], dim=1, k=2).indices.tolist())])
                stats[phase + "-type-all"] = [*stats.get(phase + "-type-all", []),
                                              [type_cor_all, preds["Type"].size()[0]]]

                # For Any Type accuracy, check if at least one index occurs in each least
                type_cor_any = sum([sum(x in p_ind for x in l_ind) > 0 for
                                    p_ind, l_ind in zip(topk(preds["Type"], dim=1, k=2).indices.tolist(),
                                                        topk(labels["Type"], dim=1, k=2).indices.tolist())])
                stats[phase + "-type-any"] = [*stats.get(phase + "-type-any", []),
                                              [type_cor_any, preds["Type"].size()[0]]]

                # Gen accuracy is the same as All Type accuracy but it's top 1 not top 2 accuracy
                # (This could be done with argmax but I was copy pasting and lazy)
                gen_cor = sum(
                    [p_ind == l_ind for p_ind, l_ind in zip(topk(preds["Gen"], dim=1, k=1).indices.tolist(),
                                                            topk(labels["Gen"], dim=1, k=1).indices.tolist())])
                stats[phase + "-gen"] = [*stats.get(phase + "-gen", []), [gen_cor, preds["Type"].size()[0]]]

                # For Shiny accuracy we just round and check equivalence since it's one number
                shiny_cor = sum([p == l for p, l in zip(round(preds["Shiny"]).tolist(), labels["Shiny"].tolist())])
                stats[phase + "-shiny"] = [*stats.get(phase + "-shiny", []), [shiny_cor, preds["Type"].size()[0]]]

        # Convert [[correct1, tested1], [correct2, tested2]] into (correct1+2)/(tested1+2) Cannot just
        # store [correct1/tested1, correct2/tested2] and average since not all batch counts are same
        for stat in stats:
            stats[stat] = sum([n[0] for n in stats[stat]]) / sum([n[1] for n in stats[stat]])
            stats_history[stat] = stats_history.get(stat, []) + [stats[stat]]

        # Print results per epoch
        print("Epoch {}/{}".format(epoch + 1, epochs) +
              "\nTrain -- Acc Type(All):{train-type-all:.4%} | Type(Any):{train-type-any:.4%} |".format(
                  **stats) +
              " Gen:{train-gen:.4%} | Shiny:{train-shiny:.4%}".format(**stats) +
              "\nTest  -- Acc Type(All):{test-type-all:.4%} | Type(Any):{test-type-any:.4%} |".format(
                  **stats) +
              " Gen:{test-gen:.4%} | Shiny:{test-shiny:.4%}".format(**stats))


if __name__ == "__main__":
    main()
