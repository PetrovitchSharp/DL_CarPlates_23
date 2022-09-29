import numpy as np
import torch
import tqdm
from torchvision import transforms

from dataset import load_dataset
from utils import collate_fn, create_model, save_model

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
    '/kaggle/input/car-plates-ocr/data',
    transformations,
    'train')
val_dataset = load_dataset(
    '/kaggle/input/car-plates-ocr/data',
    transformations,
    'val')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 2

for epoch in range(num_epochs):
    model.train()

    for images, targets in tqdm.tqdm(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    batch_losses = []
    for images, targets in tqdm.tqdm(val_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        batch_losses.append(losses.item())
        optimizer.zero_grad()

    batch_losses = np.array(batch_losses)
    batch_losses = batch_losses[np.isfinite(batch_losses)]
    print(f'Valid_loss: {np.mean(batch_losses)}')
    lr_scheduler.step()

save_model(model, 'fasterrcnn', '../models')

print("Training is finished")
