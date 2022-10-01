import argparse

import numpy as np
import torch
import tqdm
from torchvision import transforms

from dataset import load_dataset
from utils import collate_fn, create_model, save_model


def make_parser():
    parser = argparse.ArgumentParser(
        description='Train carplate model'
        )
    parser.add_argument('-data', type=str,
                        default='data',
                        help='dataset path')
    parser.add_argument('-output', type=str,
                        default='output',
                        help='output path')
    parser.add_argument('-num_epochs', type=int,
                        default=10,
                        help='train epochs')
    parser.add_argument('-batch_size', type=int,
                        default=2,
                        help='batch size')
    parser.add_argument('-exp_name', type=str,
                        default='fasterrcnn',
                        help='experiment name')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    data_path = args.data
    output_path = args.output
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    exp_name = args.exp_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model()

    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = load_dataset(
        data_path,
        transformations,
        'train')
    val_dataset = load_dataset(
        data_path,
        transformations,
        'val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        model.train()

        for images, targets in tqdm.tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device)
                       for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        batch_losses = []
        for images, targets in tqdm.tqdm(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            with torch.no_grad():
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            batch_losses.append(losses.item())

        batch_losses = np.array(batch_losses)
        batch_losses = batch_losses[np.isfinite(batch_losses)]
        print(f'Epoch: {epoch}, Valid_loss: {np.mean(batch_losses)}')
        lr_scheduler.step()

    save_model(model, f'{exp_name}_{num_epochs}.pth', output_path)

    print("Training is finished")


if __name__ == '__main__':
    main()
