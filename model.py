%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from collections import OrderedDict

alexnet = models.alexnet()
vgg16 = models.vgg16()

models = {'alexnet': alexnet, 'vgg': vgg16}

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        print("Model Created")

    def build_model(model_name, hidden_units):
        output_size = 102
        dropout_rate = 0.5

        # Check the model name to determine how model is built.
        if model_name == 'alexnet':
            print('Model is ', model_name)
            model = models[model_name]
            input_size = model.classifier[1].in_features

            for param in model.parameters():
                param.requires_grad = False

            # Create classifier for alexnet model
            classifier = nn.Sequential(OrderedDict([
                ('dropout1', nn.Dropout(p = dropout_rate)),
                ('fc1', nn.Linear(input_size, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout2', nn.Dropout(p = dropout_rate)),
                ('fc2', nn.Linear(hidden_units, output_size)),
                ('output', nn.LogSoftmax(dim=1))
                ]))

            model.classifier = classifier

            return model

        elif model_name == 'vgg':
            print('Model is ', model_name)
            model = models[model_name]
            input_size = model.classifier[0].in_features

            for param in model.parameters():
                param.requires_grad = False

            # Create classifier for vgg model
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p = dropout_rate)),
                ('fc2', nn.Linear(hidden_units, output_size)),
                ('output', nn.LogSoftmax(dim=1))
                ]))

            model.classifier = classifier

            return model

    def train_network(model, train_data, validation_data, epochs, gpu_on):
        '''
        Builds a network using feedforward and backpropagation with the VGG16 pretrained model.

        Arguments:
            model: The VGG16 pretrained model
            train_data: The training data set used to train the model
            validation_data: The validation data used to test for overfitting while training
            epochs: The amount of epochs the model will run.
            print_every: When function will print loss.
            criterion = The criterion input used for the model.
            optimizer: The optimizer input used for the model.
            device: The type of device to run the model with. Either CPU or CUDA.

        Outputs:
            Trained neural network that has been tested with validation data to prevent overfitting.
        '''
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
        device = torch.device(device)
        print("Train Network")
        steps = 0
        # Switch model dpending on user input.
        if gpu_on = False:
            model.to('cpu')
        else:
            model.to('cuda')
        # Loop through all epochs
        print('Training starting...')
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_data):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass through your model
                outputs = model.forward(inputs)

                # Find loss value
                loss = criterion(outputs, labels)
                # Backward pass though your model
                loss.backward()

                optimizer.step()
                # Add to the running loss to keep track of
                running_loss += loss.item()

                # Check the current step of epoch and print Epoch, Training Loss, Validation Loss and Test Accuracy.
                if steps % print_every == 0:
                    # Turn on evalution mode to test for overfitting.
                    model.eval()
                    # Turn off gradients for validation testing.
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, validation_data, criterion)

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.4f}.. ".format(test_loss/len(validation_data)),
                          "Test Accuracy: {:.4f}".format(accuracy/len(validation_data)))
                    # Set running loss back to 0
                    running_loss = 0
                    # Return your model to training model.
                    model.train()
        print('Training complete')

    def test_network(test_data, device):
        '''
        Tests the accuracy of the trained neural network using the test data

        Arguments:
            test_data: Tests data used to test neural network.
        Outputs:
            The accuracy of the network displayed via print.
        '''
        print("Test Network")
        correct = 0
        total = 0
        print('Testing starting...')
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                # Set images and to use cuda
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('Testing complete.')

    def save_checkpoint(classifier):
        print("Save checkpoint")
        model.class_to_idx = image_datasets['train'].class_to_idx
        checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [2509],
              'classifier': classifier,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict,
              'image_datasets': model.class_to_idx,
              'epochs': epochs}
        # Save the checkpoint
        torch.save(checkpoint, 'checkpoint.pth')

    def load_checkpoint(filepath):
        '''
        Tests the accuracy of the trained neural network using the test data

        Arguments:
            filepath: The name of the filepath where the saved model is located.
        Outputs:
            The accuracy of the network displayed via print.
        '''
        print("Load checkpoint")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def predict():
        print("Predict!")

m = Model()
