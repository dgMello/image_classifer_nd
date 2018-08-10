import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def create_train_loader(img_dir):
    '''
    Creates a train loader variable to use with network training

    Arguments:
       img_dir: The image directory where the images used for training are stored.
    Outputs:
      train_loader: The train loader used with training
      train_dataset: The dataset of images used with training
    '''
    train_img_dir = img_dir + '/train'
    # Create transforms for training data
    train_transform = transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Create dataset for training data
    train_dataset = datasets.ImageFolder(train_img_dir,
        transform=train_transform)
    # Create dataloader for training data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130,
        shuffle=True)

    return train_loader, train_dataset

def create_valid_loader(img_dir):
     '''
    Creates a valid loader variable to use with network validating

    Arguments:
       img_dir: The image directory where the images used for validating are stored.
    Outputs:
      valid_loader: The train loader used with validating
    '''
    valid_img_dir = img_dir + '/valid'
    # Create transforms for valididation data
    valid_transform = transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])
    # Create dataset for valididation data
    valid_dataset = datasets.ImageFolder(valid_img_dir,
        transform=valid_transform)
    # Create dataloader for valididation data
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=130,
        shuffle=True)
    return valid_loader


def create_test_loader(img_dir):
    '''
    Creates a test loader variable to use with network testing

    Arguments:
       img_dir: The image directory where the images used for testing are stored.
    Outputs:
      test_loader: The train loader used with testing
    '''
    test_img_dir = img_dir + '/test'
    # Create transforms for valididation data
    test_transform = transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    # Create dataset for valididation data
    test_dataset = datasets.ImageFolder(test_img_dir,
        transform=test_transform)
    # Create dataloader for valididation data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=130,
        shuffle=True)
    return test_loader


def category_mapping(json_file):
    '''
    Creates a dictionary with flower names and number used to map them.

    Arguments:
       json_file: The JSON file where the flower name and mapping info.
    Outputs:
      cat_to_name: A dictionary containing all flower names and the number they're mapped to.
    '''
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
