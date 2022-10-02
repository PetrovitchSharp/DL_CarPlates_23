import argparse
import pickle
from collections import defaultdict

import tqdm


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Train recognition model'
        )
    parser.add_argument('-output', type=str,
                        default='language_model_space.pkl',
                        help='output path')
    return parser


def main():
    # set parameters
    parser = make_parser()
    args = parser.parse_args()
    output_path = args.output
    letters = 'ABEKMHOPCTYX'
    region_numbers = []

    for i in range(1, 100):
        region_numbers.append(f'{i:02}')

    additional = [102, 113, 116, 121, 123, 124, 125, 126, 134, 136,
                  138, 142, 150, 152, 154, 159, 161, 163, 164, 173,
                  174, 177, 178, 186, 190, 196, 197, 198, 199, 277,
                  299, 716, 725, 750, 761, 763, 777, 790, 797, 799]

    for num in additional:
        region_numbers.append(str(num))

    bi_letters = [''.join([l1, l2]) for l1 in letters for l2 in letters]

    plates = []

    for l_1 in tqdm.tqdm(letters):
        for num in range(1, 1000):
            for l_23 in bi_letters:
                for region in region_numbers:
                    plates.append(
                        ''.join([l_1, f'{num:03}', l_23, region, '-'])
                        )

    language_model = defaultdict(int)
    for plate in tqdm.tqdm(plates):
        language_model[plate[0]] += 1
        language_model[plate[:2]] += 1
        for i in range(0, len(plate)-2):
            language_model[plate[i:i+3]] += 1
        for i in range(0, len(plate)-3):
            language_model[plate[i:i+4]] += 1
        for i in range(0, len(plate)-4):
            language_model[plate[i:i+5]] += 1

    with open(output_path, 'wb') as f:
        pickle.dump(language_model, f)


if __name__ == '__main__':
    main()
