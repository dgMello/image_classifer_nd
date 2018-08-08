%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

class Utilities:
    def __self__(self):
        print("Utilities started")

    def create_train_loader(train_img_dir):
        # Create transforms for training data
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              tranforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              Transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        # Create dataset for training data
        train_dataset = datasets.ImageFolder(train_img_dir, transform=train_transform)
        # Create dataloader for training data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130, shuffle=True)

        return train_loader

    def create_valid_loader(valid_img_dir):
        # Create transforms for valididation data
        valid_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              tranforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              Transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        # Create dataset for valididation data
        valid_dataset = datasets.ImageFolder(valid_img_dir, transform=train_transform)
        # Create dataloader for valididation data
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=130, shuffle=True)
        return valid_loader


   def create_test_loader(test_img_dir):
        # Create transforms for valididation data
        test_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              tranforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              Transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        # Create dataset for valididation data
        test_dataset = datasets.ImageFolder(test_img_dir, transform=train_transform)
        # Create dataloader for valididation data
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=130, shuffle=True)
        return test_loader


    def category_mapping(json_file):
        with open(json_file, 'r') as f:
            cat_to_name = json.load(f)

    return cat_to_name

u = Utilities()
