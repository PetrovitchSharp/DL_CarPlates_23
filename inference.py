import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn
import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms

from detection_utils import PlateImageAdjuster
from recognition import LanguageModel, RecognitionDataset, beam_search
from recognition_utils import Resize, collate_fn_recognition_test
from utils import load_image, load_model, load_recognition_model


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


def recognize_car_plate(img_path: Path,
                        detection_model: nn.Module,
                        recognition_model: nn.Module,
                        save_path: Path = None) -> None:
    '''
    Recognize car plate on image

    Args:
        img_path:   Path to image
        model:      Recognition model
        save_path:  Path to save image with predictions
    '''

    THRESHOLD_BOX = 0.90
    CONF_THRESHOLD = 0.201

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # noqa E501

    path_test_ocr = Path('test_ocr_data')
    path_test_ocr.mkdir(parents=True, exist_ok=True)

    normalizer = PlateImageAdjuster()

    test_plates_filenames = []

    detection_model.to(device)
    detection_model.eval()

    # We need to normalize image before passing through model
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Make predicition for one image
    img = load_image(img_path, transformations).to(device)
    with torch.no_grad():
        preds = detection_model([img])

    # Save regions with car plates for recognition

    boxes_of_interest = []
    for pred in preds:
        ps = pred['scores'].detach().cpu().numpy()
        boxes = pred['boxes'].detach().cpu().numpy()
        image = img.cpu().permute(1, 2, 0).numpy() * 255
        sorted_boxes = sorted(list(zip(ps, boxes)), key=lambda x: x[1][0]) # noqa E501
        filename = img_path.name
        n = 0
        for p, box in sorted_boxes:
            if p > THRESHOLD_BOX:
                # Too small images are useless
                if (box[2] - box[0]) * (box[3] - box[1]) < 100:
                    continue
                path = Path(filename)

                # Save bbox_image
                boxes_of_interest.append((box, p))
                bbox_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] # noqa E501
                bbox_image_file_name = ''.join(['_'.join([path.stem, str(n), 'bbox']), path.suffix]) # noqa E501
                bbox_image = normalizer(bbox_image)
                cv2.imwrite(str(path_test_ocr / bbox_image_file_name), bbox_image) # noqa E501
                plate_file_name = ''.join(['_'.join([path.stem, str(n)]), path.suffix]) # noqa E501
                test_plates_filenames.append(plate_file_name)

                # We need to save image twice (in case model returns a mask)

                plate_file_name = ''.join(['_'.join([path.stem, str(n)]), path.suffix]) # noqa E501
                cv2.imwrite(str(path_test_ocr / plate_file_name), bbox_image)
                test_plates_filenames.append(plate_file_name)

                n += 1
        if n == 0:
            j = np.argmax(ps)
            path = Path(filename)

            # Save bbox_image
            boxes_of_interest.append((box, p))
            bbox_image = image[int(boxes[j][1]):int(boxes[j][3]), int(boxes[j][0]):int(boxes[j][2])] # noqa E501
            bbox_image_file_name = ''.join(['_'.join([path.stem, str(n), 'bbox']), path.suffix]) # noqa E501
            bbox_image = normalizer(bbox_image)
            cv2.imwrite(str(path_test_ocr / bbox_image_file_name), bbox_image)
            plate_file_name = ''.join(['_'.join([path.stem, str(n)]), path.suffix]) # noqa E501
            test_plates_filenames.append(plate_file_name)

            # We need to save image twice (in case model returns a mask)
            plate_file_name = ''.join(['_'.join([path.stem, str(n)]), path.suffix]) # noqa E501
            cv2.imwrite(str(path_test_ocr / plate_file_name), bbox_image)
            test_plates_filenames.append(plate_file_name)

    with open(path_test_ocr / 'test_plates_filenames.json', 'w') as f:
        json.dump(test_plates_filenames, f)

    # Recognition
    recognition_model.to(device)
    transformations = transforms.Compose([
        Resize(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    batch_size = 64
    num_workers = 4

    test_ocr_dataset = RecognitionDataset(
        path_test_ocr,
        transformations,
        recognition_model.alphabet,
        'test')
    test_ocr_dataloader = torch.utils.data.DataLoader(
        test_ocr_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn_recognition_test)

    submission_preds = {}
    lm = LanguageModel()

    for batch in tqdm.tqdm(test_ocr_dataloader):
        with torch.no_grad():
            preds = recognition_model(batch['image'].to(device))
            preds_bbox = recognition_model(batch['image_bbox'].to(device))
        preds = preds + preds_bbox
        probs = fnn.softmax(preds, dim=2)
        preds_with_confidence = [
            beam_search(pred,
                        recognition_model.alphabet,
                        beam_width=20,
                        lm=lm,
                        alpha=0.3,
                        beta=4)
            for pred in probs.permute(1, 0, 2).cpu().data.numpy()
            ] # noqa E501
        texts_pred = [a[0] for a in preds_with_confidence]
        batch_confidence = [
            a.item()
            for a in probs.permute(1, 0, 2).std(dim=2).mean(dim=1)
            ]

        filenames = batch['file_name']
        for filename, text, conf_score in zip(filenames,
                                              texts_pred,
                                              batch_confidence):
            test_file_name, num = filename.stem.split('_')
            test_file_name = ''.join(['test/', test_file_name, filename.suffix]) # noqa E501
            if test_file_name not in submission_preds:
                submission_preds[test_file_name] = {}
            submission_preds[test_file_name][int(num)] = (text, conf_score)

    submission_dict = defaultdict(str)
    for key in submission_preds:
        sorted_keys = sorted(submission_preds[key].keys())
        if len(sorted_keys) > 1:
            submission_dict[key] = ' '.join([submission_preds[key][k][0]
                                             for k in sorted_keys if submission_preds[key][k][1] > CONF_THRESHOLD]) # noqa E501
        else:
            submission_dict[key] = submission_preds[key][sorted_keys[0]][0]
        plates = submission_dict[key].split(' ')
        for plate, box_conf in zip(plates, boxes_of_interest):
            print(f'{plate} {box_conf[0]} p={box_conf[1]}')

        if save_path is None:
            continue
        # We need to unnormalize image after passing through model
        unnormalize_1 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                             std=[1, 1, 1])
        unnormalize_2 = transforms.Normalize(mean=[0, 0, 0],
                                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]) # noqa E501
        unnormalize = transforms.Compose([unnormalize_2, unnormalize_1])
        image = unnormalize(img).cpu().numpy().transpose([1, 2, 0])
        save_result_to_jpg(image, plates, boxes_of_interest, save_path)


def save_result_to_jpg(image, plates, boxes, save_path):
    # Save result to jpg
    fig, ax = plt.subplots()
    ax.imshow(image)

    for idx in range(len(boxes)):
        bbox = boxes[idx][0]
        x0 = int(bbox[0]) - 10
        x1 = int(bbox[2]) + 10
        y0 = int(bbox[1]) - 10
        y1 = int(bbox[3]) + 10

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

        plate_num = plates[idx]
        plate_score = boxes[idx][1]

        # Adding text with licence plate and probability below bound box
        ax.text(x0, y1 + 50, f'License plate: {plate_num}, p={plate_score*100:.2f}%', color='white') # noqa E501

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
                        default='test.jpg',
                        help='path to image')
    parser.add_argument('-output', type=str,
                        default='output',
                        help='output path')
    parser.add_argument('-detection_model', type=str,
                        default='models/fasterrcnn.pth',
                        help='path to model')
    parser.add_argument('-recognition_model', type=str,
                        default='models/CRNN.pth',
                        help='path to model')
    return parser


def main() -> None:
    '''
    Main function responsible for model inference on a single image
    '''
    # Arguments parsing
    parser = make_parser()
    args = parser.parse_args()
    img_path = args.img
    output_path = args.output
    detection_model_path = args.detection_model
    recognition_model_path = args.recognition_model

    if img_path == '':
        print('You need to pass a correct image path')
    img_path = Path(img_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    detection_model = load_model(detection_model_path)
    recognition_model = load_recognition_model(recognition_model_path)
    recognize_car_plate(
        img_path,
        detection_model,
        recognition_model,
        output_path / img_path.name)


if __name__ == '__main__':
    main()
