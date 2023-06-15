from datetime import datetime as dt

import torch
from torch import optim
from torch.hub import load
from torch.optim import lr_scheduler
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose

from MLFnet import MLFnet
from utils import get_context_parts, get_backbone_layers


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(f"{device=}")

    batch_size = 16
    context = "celeba"
    base_model_name = "VGG13"

    w = load("pytorch/vision:v0.14.0", "get_model_weights", name=base_model_name)["DEFAULT"]
    model_specific_transforms = w.transforms()
    transforms = {
        'train': Compose([RandomResizedCrop(224), RandomHorizontalFlip(), model_specific_transforms]),
        'test': model_specific_transforms,
        'valid': model_specific_transforms
    }
    train_dl, test_dl, valid_dl, heads, losses, metrics = get_context_parts(context, batch_size, transforms)

    tasks = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
             "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
             "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
             "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
             "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
             "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
             "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"][:3]

    heads = {k: heads[k] for k in tasks}

    backbone, layers = get_backbone_layers(device=device)

    model = MLFnet(tasks=tuple(heads.keys()), heads=heads, backbone=backbone, device=device)

    optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.25, verbose=True)

    epochs = len(layers) * 3

    out_file = f"stats\\data-{context}-{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    with open(out_file, "w") as f:
        f.write("epoch,batch,task,task_id,param_id,param_value\n")

    for epoch in range(epochs):
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

                if phase in ["train", "test"]:
                    for task in model.tasks:
                        metrics[phase][task].update(preds[task], labels[task])

        # report metrics
        print(f"Epoch {epoch + 1}/{epochs}")
        for phase in ["train", "test"]:
            print(f"  {phase}:")
            for task in model.tasks:
                print(f"    {task}: {metrics[phase][task]}")
                metrics[phase][task].reset()


if __name__ == "__main__":
    main()
