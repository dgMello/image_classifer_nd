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
import json


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
    valid_transform = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
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
    test_transform = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
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

# Function to process an image and return a numpy array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    Arugment:
        image: image file that fill be turned into a numpy array
    Output:
        transposed_image: Image in the form of a numpy array.
    '''

    # Convert image to a pil image and get the width and height of the image.
    original_pil_image = Image.open(image)
    pil_image = original_pil_image.copy()
    width, height = pil_image.size
    size = 256

    # Check to see which is greater. Length or width. Depending on that calculate the new width and height.
    if width > height:
        ratio = float(width) / float(height)
        new_height = ratio * size
        pil_image = pil_image.resize((size, int(new_height)))
    else:
        ratio = float(height) / float(width)
        new_width = ratio * size
        pil_image = pil_image.resize((int(new_width), size))



    # Create variables for the amount to crop on each side
    width, height = pil_image.size
    crop_left = (width - 224) / 2
    crop_top = (height - 224) / 2
    crop_right = (width + 224) / 2
    crop_bottom = (height + 224) / 2

    # Crop the image with the new crop variables
    pil_image = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Convert pil image into an np array. Normalize the data.
    np_image = np.array(pil_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose the image to reorder the dimensions
    transposed_image = np_image.transpose((2, 0, 1))

    # Return you transposed image.
    return transposed_image

def predict(img, model, topk, cat_to_name, gpu_on):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.

    Arguments:
        img: The image file that will be passed through the model.
        model: The model that will be used for inference.
        topk: The number of classes the model will give a percentage for.
        cat_to_name: A dictionary containing all flower names and the number they're mapped to.
        gpu_on: Boolean value indication whether GPU will be used in prediction.
    Outputs:
        probability_chart: A chart of the class names and probabilities for each.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    img = torch.from_numpy(np.array(img)).float()
    if gpu_on == True:
        img = img.to('cuda')
    else:
        img = img.to('cpu')
    img.unsqueeze_(0)
    output = model.forward(img)
    probs, classes = torch.exp(output).topk(topk)
    # Copy the probs tensor to host memory first.
    probs = probs.cpu()
    probs = probs.detach().numpy()[0]
    # Copy the classes tensor to host memory first.
    classes = classes.cpu()
    classes = classes.numpy()[0]
    class_dict = {}
    for i in model.class_to_idx:
        class_dict[model.class_to_idx[i]] = i

    updated_classes = []
    for i in classes:
        updated_classes.append(class_dict[i])

    class_names = []
    for i in updated_classes:
        class_names.append(cat_to_name[str(i)])
    # Convert top classes and probabilites into a dataframe
    data = {'Flower Classes' : pd.Series(class_names), 'Probabilites (%)' : pd.Series(probs * 100)}
    probability_chart = pd.DataFrame(data, index = None)

    return probability_chart
