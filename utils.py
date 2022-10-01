from pathlib import Path
from typing import List

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms


def create_model() -> torch.nn.Module:
    '''
    Create customized FasterRCNN model

    Returns:
        Customized model
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    with open('../input/frcnn-model/fasterrcnn_resnet50_fpn', 'rb') as fp:
        state_dict = torch.load(fp, map_location="cpu")

    model = create_model()
    model.load_state_dict(state_dict)

    return model


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


def load_image(img_path: str, transformations: transforms.Compose) -> torch.Tensor:
    '''
    Load image from file

    Args:
        img_path:           Path to image
        transformations:    Transformations to be done with image 

    Returns:
        Loaded image
    '''
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transformations is not None:
        image = transforms(image)

    return image
