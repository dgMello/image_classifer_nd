# PROGRAMMER: DOUG MELLO
# DATE CREATED: 08/08/2018
# REVISED DATE:
# PURPOSE: Train network and save model as checkpoint.

import argparse
from os import listdir
# import utility
# from model import train_network, test_network, save_checkpoint

def main():

    in_arg = get_input_args()
    print("Command Line Arguments:\n dir=", in_arg.data_dir, "\n save directory =", in_arg.save_dir,
          "\n arch =", in_arg.arch, "\n learning rate =", in_arg.learning_rate,
          "\n hidden units =", in_arg.hidden_units,"\n epochs =", in_arg.epochs,
          "\n gpu =", in_arg.gpu)

    #trained_network = train_network()

    #test_network()

    #saved_model = save_checkpoint


def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, default = 'flowers/train', help = 'Path to the folder flowers/train.')
    parser.add_argument('--save_dir', type = str, default = 'workspace', help = 'Path to save directory.')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN model architecture to use.')
    parser.add_argument('--learning_rate', type = int, default = 0.0001, help = 'Learnrate used to train network.')
    parser.add_argument('--hidden_units', type = int, default = 2509, help = 'Number of hidden unit layers in network.')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Number of epochs used to train network.')
    parser.add_argument('--gpu', action="store_true", default = False, help = 'Turn gpu on to use for testing.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
