from pathlib import Path
from typing import List

import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.recognition import CRNN


def create_model() -> torch.nn.Module:
    '''
    Create customized FasterRCNN model

    Returns:
        Customized model
    '''
    # Device choice
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initializing model with custom prediction head
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    return model


def save_model(model: torch.nn.Module, model_name: str, path: str) -> None:
    '''
    Save trained model's weights to file

    Args:
        model:      Trained model
        model_name: Model name
        path:       Directory to save
    '''
    path = Path(path)
    path.mkdir(exist_ok=True)
    with open(path / model_name, 'wb') as fp:
        torch.save(model.state_dict(), fp)


def load_model(path: str) -> torch.nn.Module:
    '''
    Load trained model from file

    Args:
        path: Path to model's weights

    Returns:
        FasterRCNN model with loaded weights
    '''
    with open(path, 'rb') as fp:
        state_dict = torch.load(fp, map_location="cpu")

    model = create_model()
    model.load_state_dict(state_dict)

    return model


def load_recognition_model(path: str) -> torch.nn.Module:
    '''
    Load trained model from file

    Args:
        path: Path to model's weights

    Returns:
        RCNN model with loaded weights
    '''
    with open(path, 'rb') as fp:
        state_dict = torch.load(fp, map_location="cpu")
    crnn = CRNN(rnn_bidirectional=True)
    crnn.load_state_dict(state_dict)

    return crnn


def collate_fn(samples: List) -> tuple:
    '''
    Collate lists of samples into batches

    Args:
        samples: List of samples

    Returns:
        Batch
    '''
    return tuple(zip(*samples))


def detach_dict(pred: dict) -> dict:
    '''
    Detach dict

    Args:
        Prediction of a model

    Returns:
        Detached dict
    '''
    return {k: v.detach().cpu() for (k, v) in pred.items()}


def load_image(img_path: Path, transformations: transforms.Compose
               ) -> torch.Tensor:
    '''
    Load image from file

    Args:
        img_path:           Path to image
        transformations:    Transformations to be done with image 

    Returns:
        Loaded image
    '''
    image = cv2.imread(str(img_path))
    # conversion from BGR to RGB color space 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Additional transformations
    if transformations is not None:
        image = transformations(image)

    return image
