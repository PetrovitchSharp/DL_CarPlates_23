import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fnn
import tqdm
from Levenshtein import distance
from torchvision import transforms

from src.recognition import CRNN, RecognitionDataset
from src.recognition_utils import Resize, collate_fn_recognition, decode


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Train recognition model'
        )
    parser.add_argument('-data', type=str,
                        default='ocr_data',
                        help='dataset path')
    parser.add_argument('-output', type=str,
                        default='models',
                        help='output path')
    parser.add_argument('-num_epochs', type=int,
                        default=20,
                        help='train epochs')
    parser.add_argument('-batch_size', type=int,
                        default=64,
                        help='batch size')
    parser.add_argument('-exp_name', type=str,
                        default='CRNN',
                        help='experiment name')
    return parser


def main():
    # set parameters
    parser = make_parser()
    args = parser.parse_args()
    data_path = args.data
    output_path = args.output
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    experiment_name = args.exp_name
    num_workers = 4

    # Create model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # noqa E501
    crnn = CRNN(rnn_bidirectional=True)
    crnn.to(device)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(crnn.parameters(),
                                 lr=3e-4,
                                 amsgrad=True,
                                 weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=1/np.sqrt(10),
        patience=2,
        verbose=True,
        threshold=1e-3)

    # Create datasets and dataloaders
    transformations = transforms.Compose([
        Resize(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ocr_dataset = RecognitionDataset(
        data_path,
        transformations,
        crnn.alphabet,
        'train'
        )
    val_ocr_dataset = RecognitionDataset(
        data_path,
        transformations,
        crnn.alphabet,
        'val')

    train_dataloader = torch.utils.data.DataLoader(
        train_ocr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_recognition)
    val_dataloader = torch.utils.data.DataLoader(
        val_ocr_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_recognition)

    # Train model
    best_loss = np.inf
    prev_lr = optimizer.param_groups[0]['lr']

    for i in range(num_epochs):
        epoch_losses = []
        levensteint_losses = []

        # Если поменялась lr - загружаем лучшую модель
        if optimizer.param_groups[0]['lr'] < prev_lr:
            prev_lr = optimizer.param_groups[0]['lr']
            with open(output_path / f'{experiment_name}.pth', 'rb') as fp:
                state_dict = torch.load(fp, map_location="cpu")
            crnn.load_state_dict(state_dict)
            crnn.to(device)

        crnn.train()
        for b in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = fnn.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int() # noqa E501

            texts_pred = decode(seqs_pred, crnn.alphabet)
            texts_gt = b["text"]
            levensteint_losses.extend([distance(pred, gt)
                                       for pred, gt in zip(texts_pred,
                                                           texts_gt)
                                       ])

            loss = fnn.ctc_loss(log_probs=log_probs,  # (T, N, C)
                                targets=seqs_gt,  # N, S or sum(target_lengths)
                                input_lengths=seq_lens_pred,  # N
                                target_lengths=seq_lens_gt)  # N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        print(f'Train {i + 1}, {np.mean(epoch_losses)}')
        print(f'Train {i + 1} Levenstein, {np.mean(levensteint_losses)}')
        time.sleep(0.5)

        epoch_losses = []
        levensteint_losses = []
        crnn.eval()
        for b in tqdm.tqdm(val_dataloader, total=len(val_dataloader)):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = fnn.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int() # noqa E501

            texts_pred = decode(seqs_pred, crnn.alphabet)
            texts_gt = b["text"]
            levensteint_losses.extend([distance(pred, gt)
                                       for pred, gt in zip(texts_pred,
                                                           texts_gt)
                                       ])

            loss = fnn.ctc_loss(log_probs=log_probs,  # (T, N, C)
                                targets=seqs_gt,  # N, S or sum(target_lengths)
                                input_lengths=seq_lens_pred,  # N
                                target_lengths=seq_lens_gt)  # N

            epoch_losses.append(loss.item())

            if best_loss > epoch_losses[-1]:
                best_loss = epoch_losses[-1]
                with open(output_path / f'{experiment_name}.pth', 'wb') as fp:
                    torch.save(crnn.state_dict(), fp)

        lr_scheduler.step(np.mean(levensteint_losses))
        print(f'Valid {i + 1}, {np.mean(epoch_losses)}')
        print(f'Valid {i + 1} Levenstein, {np.mean(levensteint_losses)}')
        time.sleep(0.5)


if __name__ == '__main__':
    main()
