import argparse
from pathlib import Path
from typing import List

import easyocr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from utils import detach_dict, load_image, load_model


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


def recognize_car_plate(img_path: Path, model: nn.Module,
                        save_path: Path) -> None:
    '''
    Recognize car plate on image

    Args:
        img_path:   Path to image
        model:      Recognition model
        save_path:  Path to save image with predictions
    '''
    model.cpu()
    model.eval()

    # We need to normalize image before passing through model
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = load_image(img_path, transformations)

    predictions = model([img])
    predictions = [detach_dict(pred) for pred in predictions]

    boxes = get_boxes(predictions, 0.9)
    # We can have several car plates in an image
    plates_count = len(boxes)

    # We need to unnormalize image after passing through model
    unnormalize_1 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1, 1, 1])
    unnormalize_2 = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    unnormalize = transforms.Compose([unnormalize_2, unnormalize_1])

    for idx, box in enumerate(boxes):
        # Slight expansion of the frame for ease of recognition
        x0 = int(box[0]) - 20
        x1 = int(box[2]) + 20
        y0 = int(box[1]) - 20
        y1 = int(box[3]) + 20

        # Image cropping
        im = unnormalize(img).numpy().transpose([1, 2, 0])[y0:y1, x0:x1]

        # Temporary saving of an image
        # It is more convenient to feed saved images into the model than to convert the tensor to the desired format )))
        plt.imsave(f'plate_{idx}.jpg', im)

    detections = []

    reader = easyocr.Reader(['ru', 'en'])

    for idx in range(plates_count):
        detections.append(
            reader.readtext(
                f'plate_{idx}.jpg',
                decoder='beamsearch',
                allowlist=' УКЕНВАРОСМИТХ1234567890'))
        # Deleting temporary file
        Path(f'plate_{idx}.jpg').unlink()

    # We don't need to plot an image, just add boxes and text and save 
    plt.ioff()
    plt.close()

    fig, ax = plt.subplots()

    image = unnormalize(img)

    ax.imshow(image.numpy().transpose([1, 2, 0]))

    for idx in range(plates_count):
        x0 = int(boxes[idx][0]) - 10
        x1 = int(boxes[idx][2]) + 10
        y0 = int(boxes[idx][1]) - 10
        y1 = int(boxes[idx][3]) + 10

        # Adding bound box to an image
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

        # Adding text with licence plate and probability below bound box
        ax.text(x0, y1 + 50, f'License plate: {plate_num}', color='white')
        ax.text(
            x0,
            y1 + 110,
            f'Probability: {plate_score*100:.2f} %',
            color='white')

    print(f'License plate: {plate_num}, p={plate_score*100:.2f}%')

    # Saving image with detections
    fig.savefig(save_path)


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Train carplate model'
        )
    parser.add_argument('-img', type=str,
                        default='',
                        help='path to image')
    parser.add_argument('-output', type=str,
                        default='output',
                        help='output path')
    parser.add_argument('-model', type=str,
                        default='',
                        help='path to model')
    return parser


def main() -> None:
    '''
    Main function responsible for model inference on a single image
    '''
    #region Arguments parsing
    parser = make_parser()
    args = parser.parse_args()
    img_path = args.img
    output_path = args.output
    model_path = args.model
    #endregion

    if img_path == '':
        print('You need to pass a correct image path')
    img_path = Path(img_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    model = load_model(model_path)
    recognize_car_plate(img_path, model, output_path / img_path.name)


if __name__ == '__main__':
    main()
