import torch
from torch import tensor, optim
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
        with open("data/multimon/type_weights.txt", "r") as f:
            type_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            type_indexes = {line.split(":")[0]: index for index, line in enumerate(type_counts)}

        with open("data/multimon/gen_weights.txt", "r") as f:
            gen_counts = {line.split(":")[0]: int(line.split(":")[1].strip()) for line in f}
            gen_indexes = {line.split(":")[0]: index for index, line in enumerate(gen_counts)}

        train_dataset = MultimonDataset(data_file=f"./data/{context}/labels.txt", device=device,
                                        type_dict=type_indexes, gen_dict=gen_indexes,
                                        key_mask=partitions["train"], img_path=f"./data/{context}/data",
                                        random_transforms=2, random_transforms_list=random_transforms_list)
        test_dataset = MultimonDataset(data_file=f"./data/{context}/labels.txt", device=device,
                                       type_dict=type_indexes, gen_dict=gen_indexes,
                                       key_mask=partitions["test"], img_path=f"./data/{context}/data",
                                       random_transforms=2, random_transforms_list=random_transforms_list)

        heads = {"Type": Sequential(Flatten(),
                                    LazyLinear(out_features=1024), ReLU(),
                                    Linear(in_features=1024, out_features=1024), ReLU(),
                                    Linear(in_features=1024, out_features=len(type_counts))),
                 "Gen": Sequential(Flatten(),
                                   LazyLinear(out_features=1024), ReLU(),
                                   Linear(in_features=1024, out_features=1024), ReLU(),
                                   Linear(in_features=1024, out_features=len(gen_counts))),
                 "Shiny": Sequential(LazyLinear(out_features=1024), ReLU(),
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

    return train_dataloader, test_dataloader, heads, losses


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(f"{device=}")

    train_dataloader, test_dataloader, heads, losses = get_context_parts("celeba", device, 32)

    model = MLFnet(tasks=tuple(heads.keys()), heads=heads, device=device)

    quit()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


if __name__ == "__main__":
    main()
