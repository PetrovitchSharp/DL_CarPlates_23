from typing import List

import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import load_model, load_image, detach_dict


def get_boxes(predictions: List, threshold: float) -> List[torch.Tensor]:
    '''
    Get boundboxes of detected car plates

    Args:
        predictions:    Model preditions for the image
        threshold:      Score threshold

    Returns:
        List of boundboxes of detected car plates
    '''
    return predictions[0]['boxes'][predictions[0]['scores'] > threshold]


def recognize_car_plate(img_path: str, model: nn.Module,
                        save_path: str) -> None:
    '''
    Recognize car plate on image

    Args:
        img_path:   Path to image
        model:      Recognition model
        save_path:  Path to save image with predictions
    '''
    model.cpu()
    model.eval()

    img = load_image(img_path)

    predictions = model([img])
    predictions = [detach_dict(pred) for pred in predictions]

    boxes = get_boxes(predictions, 0.9)
    plates_count = len(boxes)

    unnormalize_1 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1, 1, 1])
    unnormalize_2 = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    unnormalize = transforms.Compose([unnormalize_2, unnormalize_1])

    for idx, box in enumerate(boxes):
        x0 = int(box[0]) - 20
        x1 = int(box[2]) + 20
        y0 = int(box[1]) - 20
        y1 = int(box[3]) + 20

        im = unnormalize(img).numpy().transpose([1, 2, 0])[y0:y1, x0:x1]
        plt.imsave(f'plate_{idx}.jpg', im)

    detections = []

    reader = easyocr.Reader(['ru', 'en'])

    for idx in range(plates_count):
        detections.append(
            reader.readtext(
                f'plate_{idx}.jpg',
                decoder='beamsearch',
                allowlist=' УКЕНВАРОСМИТХ1234567890'))

    plt.close()

    fig, ax = plt.subplots()

    image = unnormalize(img)

    for idx in range(plates_count):
        x0 = int(boxes[idx][0]) - 10
        x1 = int(boxes[idx][2]) + 10
        y0 = int(boxes[idx][1]) - 10
        y1 = int(boxes[idx][3]) + 10

        rect = patches.Rectangle(
            (x0,
             y0),
            x1 - x0,
            y1 - y0,
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)

        plate_num = detections[idx][0][1]
        plate_score = detections[idx][0][2]

        ax.text(x0, y1 + 50, f'License plate: {plate_num}', color='white')
        ax.text(
            x0,
            y1 + 110,
            f'Probability: {plate_score*100:.2f} %',
            color='white')

    print(f'License plate: {plate_num}')
    print(f'Probability: {plate_score*100:.2f} %')

    fig.savefig(save_path)
