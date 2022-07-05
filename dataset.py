import os
from random import sample

import torch
from matplotlib import pyplot as plt
from torch import zeros, tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, \
    RandomOrder, Compose
from torchvision.transforms.functional import resize
from tqdm import tqdm


class MLFnetDataset(Dataset):
    def __init__(self, data_file, key_mask, img_path, device, no_mask=False,
                 random_transforms=0, random_transforms_list=None):
        self.img_path = img_path
        self.device = device

        self.random_transforms_list = random_transforms_list
        transform_list_len = len(self.random_transforms_list) if self.random_transforms_list is not None else 0
        self.random_transforms = min(random_transforms, transform_list_len)
        self.random_transforms = max(-1, self.random_transforms)
        # random transforms
        # Value   | -1                | 0            | 0<n<=len(transforms)
        # Result  | All in rand order | All in order | Randomly pick n to apply

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]
        img_path = os.path.join(self.img_path, key)
        image = read_image(img_path).float().to(self.device)
        image /= 255
        labels = self.data[key]

        if self.random_transforms_list:
            if self.random_transforms > 0:  # only self.random_transforms > 0 requires filtering
                transform_list = sample(self.random_transforms_list, self.random_transforms)
            else:
                transform_list = self.random_transforms_list
            if self.random_transforms == 0:  # only self.random_transforms == 0 requires in order
                transforms = Compose(transform_list)
            else:
                transforms = RandomOrder(transform_list)
            image = transforms(image)

        image = resize(image, [64, 64])

        return image, labels

    # def get_size(self, idx):
    #     img, _ = self.__getitem__(idx)
    #     print(img.size())


class MultimonDataset(MLFnetDataset):
    def __init__(self, data_file, type_dict, gen_dict, key_mask, img_path, device, no_mask=False, random_transforms=0,
                 random_transforms_list=None, load_fraction=1):
        # TODO add on the fly index masking to enable k fold
        super().__init__(data_file, key_mask, img_path, device, no_mask, random_transforms, random_transforms_list)

        self.no_mask = no_mask
        key_mask = key_mask[:int(len(key_mask) * load_fraction)]
        # loads image key and targets
        self.data = self.parse_datafile(data_file, key_mask, type_dict=type_dict, gen_dict=gen_dict)

    def parse_datafile(self, data_path, key_filter, type_dict, gen_dict, tqdm_on=True):
        filter = dict.fromkeys(key_filter, True)  # using a filter dict saves iterating over the filters
        data = dict()
        with open(data_path) as file:
            if tqdm_on:
                lines = tqdm(file)
                lines.set_description("Loading dataset... ")
            else:
                lines = file

            for line in lines[1:]:
                splits = line.split(",")
                if filter.get(splits[0], False) or self.no_mask:
                    d1 = {"Type": zeros(len(type_dict)).scatter_(0, tensor([type_dict[s] for s in splits[1:3]]), 1),
                          "Gen": zeros(len(gen_dict)).scatter_(0, tensor([gen_dict[splits[3]]]), 1)}
                    d2 = {task.title(): tensor(splits[i + 4]).float() for i, task in
                          enumerate(["hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"])}
                    data[splits[0]] = {**d1, **d2}

        return data


class CelebADataset(MLFnetDataset):
    def __init__(self, data_file, key_mask, img_path, device, no_mask=False, random_transforms=0,
                 random_transforms_list=None, load_fraction=1):
        # TODO add on the fly index masking to enable k fold
        super().__init__(data_file, key_mask, img_path, device, no_mask, random_transforms, random_transforms_list)

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

    with open("data/multimon/gen_weights.txt", "r") as f:
        gens = {line.split(":")[0]: index for index, line in enumerate(f)}
    print(gens)
    with open("data/multimon/type_weights.txt", "r") as f:
        types = {line.split(":")[0]: index for index, line in enumerate(f)}
    print(types)

    index_len = len(open("data.txt").readlines())

    training_index_mask = sample([i for i in range(index_len)], int(index_len * 0.8))
    testing_index_mask = [i for i in range(index_len) if i not in training_index_mask]

    # training_data = MultimonDataset(data_file="data.txt", type_dict=types, gen_dict=gens,
    #                                 index_mask=training_index_mask, img_path="./sprites/processed")
    # testing_data = MultimonDataset(data_file="data.txt", type_dict=types, gen_dict=gens,
    #                                index_mask=testing_index_mask, img_path="./sprites/processed")

    random_transforms_list = [RandomResizedCrop(size=96, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)),
                              RandomHorizontalFlip(1),  # Force a flip if selected
                              RandomVerticalFlip(1),
                              RandomAffine(degrees=45),  # picks an angle between -45 and +45
                              RandomAffine(degrees=0, shear=[-30, 30, 0, 0]),  # X shear
                              RandomAffine(degrees=0, shear=[0, 0, -30, 30]),  # X shear
                              RandomAffine(degrees=0, translate=[0.2, 0.2])  # X shear
                              ]

    random_data = MultimonDataset(data_file="data.txt", type_dict=types, gen_dict=gens, device=device,
                                  key_mask=[], no_mask=True, img_path="./sprites/processed",
                                  random_transforms_list=random_transforms_list, random_transforms=2)

    for data, labels in random_data:
        print("Here!")
        print(data.permute(1, 2, 0).size())
        plt.imshow(data.permute(1, 2, 0))
        plt.show()
        input()
    #
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
