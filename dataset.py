import os
from pprint import pprint

import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from torch import zeros, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import RandomResizedCrop, Compose
from torchvision.transforms.functional import resize
from tqdm import tqdm


class MLFnetDataset(Dataset):
    def __init__(self, data_file, key_mask, img_path, device, no_mask=False, transforms=None, output_size=None):
        self.img_path = img_path
        self.device = device
        self.transforms = transforms
        self.output_size = output_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]

        img_path = os.path.join(self.img_path, key)
        image = read_image(img_path).float().to(self.device)
        if self.transforms is not None:
            image = self.transforms(image)
        if self.output_size is not None:
            image = resize(image, self.output_size)

        labels = self.data[key]

        return image, labels

    # def get_size(self, idx):
    #     img, _ = self.__getitem__(idx)
    #     print(img.size())


class MultimonDataset(Dataset):
    def __init__(self, data_file, part_file, img_path, device=None, transforms=None,
                 partition="train", output_size=64):
        if data_file is None or part_file is None or img_path is None:
            raise ValueError("data_file, part_file, and img_path must be specified")

        if partition not in ["train", "test", "val"]:
            raise ValueError("partition must be one of 'train', 'test', or 'val'")
        else:
            partition = ["train", "test", "val"].index(partition)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_file = data_file
        self.img_path = img_path
        self.device = device
        self.transforms = transforms
        self.output_size = output_size
        self.type_counts = 0
        self.gen_counts = 0

        self.type_weights = None
        self.gen_weights = None

        # load partitions
        self.partitions = self.parse_partitions(part_file, partition)

        # loads image key and targets
        self.data = self.parse_datafile(data_file)

    def parse_partitions(self, part_file, partition):
        parts = read_csv(os.path.join(self.img_path, "..", part_file))
        filter = parts["split"] == partition
        return parts["index"][filter]

    def parse_datafile(self, data_path, data_format="std"):
        data = read_csv(os.path.join(self.img_path, "..", data_path))

        self.type_counts = max(data["type1"].max(), data["type2"].max()) + 1
        self.gen_counts = data["gen"].max() + 1

        temp1 = data.value_counts("type1").to_dict()
        temp2 = data.value_counts("type2").to_dict()
        self.type_weights = {t: (temp1.get(t, 0) + temp2.get(t, 0)) / (2 * len(data))
                             for t in list(temp1.keys()) + list(temp2.keys())}

        self.gen_weights = data.value_counts("gen").to_dict()

        stats = ["hp", "att", "def", "spatt", "spdef", "speed"]

        filter = data["index"].isin(self.partitions)
        data = data[filter]

        data = {row["index"]: {"type": zeros(self.type_counts).scatter_(0, tensor([row["type1"], row["type2"]]), 1),
                               "gen": zeros(self.gen_counts).scatter_(0, tensor([row["gen"]]), 1),
                               **{x: tensor(row[x]).float() for x in stats}}
                for index, row in data.iterrows()}

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]

        img_path = os.path.join(self.img_path, key)
        image = read_image(img_path).float().to(self.device) / 255.0
        if self.transforms is not None:
            image = self.transforms(image)
        if self.output_size is not None:
            image = resize(image, [self.output_size])

        labels = self.data[key]

        return image, labels


class CelebADataset(MLFnetDataset):
    def __init__(self, data_file, key_mask, img_path, device, no_mask=False, transforms=None,
                 output_size=None, load_fraction=1):
        # TODO add on the fly index masking to enable k fold
        super().__init__(data_file, key_mask, img_path, device, no_mask, transforms, output_size)

        self.tasks = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                      "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                      "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                      "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                      "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
                      "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                      "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

        self.no_mask = no_mask
        key_mask = key_mask[:int(len(key_mask) * load_fraction)]
        self.data = self.parse_datafile(data_file, key_mask)  # loads image key and targets

    def parse_datafile(self, data_path, key_filter, tqdm_on=True):
        filter = dict.fromkeys(key_filter, True)  # using a filter dict saves iterating over the filters
        data = dict()
        with open(data_path) as file:
            if tqdm_on:
                lines = tqdm(file)
                lines.set_description("Loading dataset... ")
            else:
                lines = file

            for line in lines:
                splits = line.split(",")
                if filter.get(splits[0], False) or self.no_mask:
                    data[splits[0]] = {task: value.unsqueeze(0) for task, value in
                                       zip(self.tasks, [tensor(int(s)).float() for s in splits[1:]])}
        return data


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(device)

    transforms = Compose([RandomResizedCrop(64)])

    random_data = MultimonDataset(data_file="data.csv", part_file="partitions.csv", img_path="./data/multimon/sprites",
                                  device=device, transforms=transforms, partition="train")

    loader = DataLoader(random_data, batch_size=1, shuffle=True)

    for data, labels in loader:
        data = data.cpu()[0]
        print(data.permute(1, 2, 0).size())
        plt.imshow(data.permute(1, 2, 0))
        plt.show()
        print("Here!")
        pprint(labels)
        input("Enter to continue...")
