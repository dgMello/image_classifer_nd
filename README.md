# Image Classifier Project


## Description
  This app allows the user to train a neural network to learn differnt plant
  types using pictures of them.
  The app will then be able to be fed a picture a one of the trained flowers and
  predict which flower it is.

## Installation
  Part 1: Training neural network.

    $ git clone https://github.com/dgMello/image_classifer_nd.

    Open command prompt.

    Navigate to where repository is.

    Enter the following into command prompt with arguments below: $ python train.py.

    --data_dir: Location of data directory where pictures used to train network are located. Required

    --save_dir: Where checkpoint will be save. Will default to current directory 6is no argument is provided.

    --arch: Must choose 'alexnet' or 'vgg16'.

    --learning_rate: Learnrate used to train network. Defaults to 0.0001 if no argument is provided.

    --hidden_units: Number of hidden unit layers in network. Defaults to 2509 if no argument is provided.

    --epochs: Number of epochs used to train network. Defaults to 3 if no argument is provided.

    --gpu: If entered GPU will be used to train network. Defaults to False if no argument is provided.

Part 2: Predicting with trained neural network

    Enter the following into command prompt with arguments below: $ python predict.py.

    input: Filepath of picture that will be inputed into the neural network. Required.

    checkpoint: Filepath of checkpoint file that will be load neural network.

    --top_k: Number of classes neural network will provide a prediction. Defaults to 5 if no argument is provided.

    --category_names: File that contains categories mapping. Defaults to cat_to_name.json if no argument is provided.

    --gpu: If entered GPU will be used to do prediction. Defaults to False if no argument is provided.

## Usage
  Python 3.6.5 or later is required.

## Programming languages used
  Python

## Python packages used
  Pandas

  Matplotlib

  NumPy

  PyTorch

## Third party web applications used
  Jupyter Notebook

## Third party resources.

https://stackoverflow.com

https://www.w3schools.com


## Author: Doug Mello

## Version: 1.0
