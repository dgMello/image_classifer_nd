# PROGRAMMER: DOUG MELLO
# DATE CREATED: 08/08/2018
# REVISED DATE:
# PURPOSE: Load a checkpoint and predict an image provided by user.

import argparse
from utility import category_mapping
from os import listdir
from model import Model

def main():
    print('Predict')
    in_arg = get_input_args()
    print("Command Line Arguments:\n input =", in_arg.input, "\n checkpoint =",
        in_arg.checkpoint, "\n top_k =", in_arg.top_k, "\n category_names =",
        in_arg.category_names, "\n gpu =", in_arg.gpu)


def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str,
        default='flowers/train/12/image_03995.jpg',
        help = 'Path to the image file that will be predicted. .')
    parser.add_argument('checkpoint', type = str, default = 'checkpoints.pth',
        help = 'Path to checkpoint file.')
    parser.add_argument('--top_k', type = int, default = 5,
        help = 'Number of most likley classes.')
    parser.add_argument('--category_names', type = str,
        default = 'cat_to_name.json',
        help = 'File that contains categories mapping.')
    parser.add_argument('--gpu', action="store_true", default = False,
        help = 'Turn gpu on to use for inference.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
