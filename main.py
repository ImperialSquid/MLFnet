from datetime import datetime as dt

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, Normalize, Resize, \
    CenterCrop

from MLFnet import MLFnet
from utils import get_context_parts


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(f"{device=}")

    batch_size = 16
    context = "celeba"
    input_size = 256
    transforms = {
        'train': Compose([RandomResizedCrop(input_size), RandomHorizontalFlip(),  # slight randomness for training
                          ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': Compose([Resize(input_size), CenterCrop(input_size),  # no randomness for testing
                         ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': Compose([Resize(input_size), CenterCrop(input_size),  # no randomness for validation
                          ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    train_dl, test_dl, valid_dl, heads, losses, get_acc = \
        get_context_parts(context, device, batch_size, transforms)

    backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

    layers = [{"type": "Linear", "in_channels": 3, "out_channels": 6}]

    model = MLFnet(tasks=tuple(heads.keys()), heads=heads, device=device, backbone=None)
    model.add_layer(None, type="Flatten")
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
