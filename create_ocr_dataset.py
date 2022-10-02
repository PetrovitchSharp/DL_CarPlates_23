# Extract car plates from raw dataset for OCR model training

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tqdm

from detection_utils import (PlateImageAdjuster, PlateImageExtractor,
                             build_mask, get_rectangular_box)
from recognition_utils import normalize_text


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
                        default='data',
                        help='data path')
    parser.add_argument('-output', type=str,
                        default='ocr_data',
                        help='output path')
    return parser


def main():
    # set parameters
    parser = make_parser()
    args = parser.parse_args()
    output_path = args.output
    path_data = Path(args.data)
    path_ocr_dataset = Path(output_path)
    path_ocr_dataset.mkdir(parents=True, exist_ok=True)

    plates_filename = path_data / 'train.json'
    with open(plates_filename) as f:
        json_data = json.load(f)

    normalizer = PlateImageAdjuster()
    extractor = PlateImageExtractor()

    '''For each image from train dataset extract car plate and save to
    corresponding file '''
    for sample in tqdm.tqdm(json_data):
        if sample['file'] == 'train/25632.bmp':
            continue
        file_path = path_data / sample['file']
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for plate in sample['nums']:
            box = plate['box']
            text = plate['text']
            mask = build_mask(box, image)
            plate_img = extractor(image, mask, np.array(box))
            plate_img = normalizer(plate_img)
            text = normalize_text(text)
            file_path = path_ocr_dataset / ''.join([text, '.png'])
            cv2.imwrite(str(file_path), plate_img)

            # save also bboxes
            file_path = path_ocr_dataset / ''.join([text, '_bbox.png'])
            raw_box = get_rectangular_box(box)
            plate_bbox = image[raw_box[1]:raw_box[3], raw_box[0]:raw_box[2], :]
            plate_bbox = normalizer(plate_bbox)
            cv2.imwrite(str(file_path), plate_bbox)


if __name__ == '__main__':
    main()
