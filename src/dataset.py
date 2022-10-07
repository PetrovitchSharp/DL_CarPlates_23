import json
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms

from src.utils import load_image


class CarPlatesDatasetWithRectangularBoxes(data.Dataset):
    '''
    Custom dataset to solve car plates recognition task
    '''

    def __init__(self, root: str, transforms: transforms.Compose, mode: str,
                 train_size: int = 0.9) -> None:
        '''
        Args:
            root:       Path to directory with dataset
            transforms: Transforms to be done with images
            mode:       Type of dataset
            train_size: Size of train part
        '''
        super(CarPlatesDatasetWithRectangularBoxes, self).__init__()
        self.root = Path(root)
        self.train_size = train_size

        self.image_names = []
        self.image_ids = []
        self.image_boxes = []
        self.image_texts = []
        self.box_areas = []

        self.transforms = transforms

        # Loading of train data
        plates_filename = self.root / 'train.json'
        with open(plates_filename) as f:
            json_data = json.load(f)

        # Train\Validation split
        train_valid_border = int(len(json_data) * train_size) + 1
        data_range = (0, train_valid_border) if mode == 'train' \
            else (train_valid_border, len(json_data))
        
        self.load_data(json_data[data_range[0]:data_range[1]])
        return

    def load_data(self, json_data: dict) -> None:
        '''
        Load data from json with data markup

        Args:
            json_data: Object with dataset markup
        '''
        for i, sample in enumerate(json_data):
            if sample['file'] == 'train/25632.bmp':
                continue
            self.image_names.append(self.root / sample['file'])
            self.image_ids.append(torch.Tensor([i]))

            boxes = []
            texts = []
            areas = []

            for box in sample['nums']:
                # Getting bound box of car plate
                points = np.array(box['box'])
                x_0 = np.min([points[0][0], points[3][0]])
                y_0 = np.min([points[0][1], points[1][1]])
                x_1 = np.max([points[1][0], points[2][0]])
                y_1 = np.max([points[2][1], points[3][1]])

                # Coordinates are mixed up in some files
                if x_0 > x_1:
                    x_1, x_0 = x_0, x_1
                if y_0 > y_1:
                    y_1, y_0 = y_0, y_1

                boxes.append([x_0, y_0, x_1, y_1])

                texts.append(box['text'])
                areas.append(np.abs(x_0 - x_1) * np.abs(y_0 - y_1))

            boxes = torch.FloatTensor(boxes)
            areas = torch.FloatTensor(areas)

            self.image_boxes.append(boxes)
            self.image_texts.append(texts)
            self.box_areas.append(areas)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        '''
        Get image from dataset

        Args:
            idx: Index of image in dataset
        '''
        target = {}

        # Filling target dict with information about image
        if self.image_boxes is not None:
            boxes = self.image_boxes[idx].clone()
            areas = self.box_areas[idx].clone()

            num_boxes = boxes.shape[0]

            target['boxes'] = boxes
            target['area'] = areas
            target['labels'] = torch.LongTensor([1] * num_boxes)
            target['image_id'] = self.image_ids[idx].clone()
            target['iscrowd'] = torch.Tensor([False] * num_boxes)

        # Loading image itself
        image = load_image(str(self.image_names[idx]), self.transforms)
        
        return image, target

    def __len__(self) -> int:
        '''
        Get size of dataset

        Returns:
            Size of custom dataset
        '''
        return len(self.image_names)


def load_dataset(
        path: str, transformations: transforms.Compose, mode: str) -> CarPlatesDatasetWithRectangularBoxes:
    '''
    Load images and create dataset containing them

    Args:
        path:               Path to directory with images
        transformations:    Transormations to be done with images
        mode:               Type of dataset

    Returns:
        Custom dataset with images in it
    '''
    return CarPlatesDatasetWithRectangularBoxes(path, transformations, mode)
