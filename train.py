# PROGRAMMER: DOUG MELLO
# DATE CREATED: 08/08/2018
# REVISED DATE:
# PURPOSE: Train network and save model as checkpoint.

import argparse
from utility import create_train_loader, create_valid_loader, create_test_loader
from os import listdir
from model import Model

def main():
    print("Starting main function")
    in_arg = get_input_args()
    print("Command Line Arguments:\n dir =", in_arg.data_dir,
        "\n save directory =", in_arg.save_dir, "\n arch =", in_arg.arch,
        "\n learning rate =", in_arg.learning_rate, "\n hidden units =",
        in_arg.hidden_units,"\n epochs =", in_arg.epochs,
        "\n gpu =", in_arg.gpu)

    new_model = Model(in_arg.arch)
    # Call create_train_loader to create train loader.
    train_loader, train_dataset = create_train_loader(in_arg.data_dir)
    # Call create_valid_loader to create valid loader.
    valid_loader = create_valid_loader(in_arg.data_dir)
    # Call build_model to create model.
    built_model, input_size = new_model.build_model(in_arg.hidden_units)

    optimizer = new_model.train_network(built_model, train_loader, valid_loader,
        in_arg.learning_rate, in_arg.epochs, in_arg.gpu)

    # Call create_test_loader to create test loader.
    test_loader = create_test_loader(in_arg.data_dir)
    # Test the newly trained network by calling the test_network function
    new_model.test_network(built_model, test_loader, in_arg.gpu)
    # Save the training image dataset to your model.
    built_model.class_to_idx = train_dataset.class_to_idx
    # Save model by calling the save_checkpoint fucntion
    saved_model = new_model.save_checkpoint(built_model, input_size,
        in_arg.hidden_units, optimizer, built_model.class_to_idx, in_arg.epochs)


def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default='flowers', type = str,
        help = 'Path to the folder flowers.')
    parser.add_argument('--save_dir', type = str, default = 'workspace',
        help = 'Path to save directory.')
    parser.add_argument('--arch', type = str, choices = ('vgg16', 'alexnet'),
        required=True, help = 'CNN model architecture to use.')
    parser.add_argument('--learning_rate', type = int, default = 0.0001,
        help = 'Learnrate used to train network.')
    parser.add_argument('--hidden_units', type = int, default = 2509,
        help = 'Number of hidden unit layers in network.')
    parser.add_argument('--epochs', type = int, default = 3,
        help = 'Number of epochs used to train network.')
    parser.add_argument('--gpu', action="store_true", default = False,
        help = 'Turn gpu on to use for testing.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
