import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from collections import OrderedDict

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'alexnet': alexnet, 'vgg16': vgg16}

class Model:
    def __init__(self, model_type):
        self.model_type = model_type
        print("Model Created")

    def build_model(self, hidden_units):
        '''
        Creates a model with the pretrained network from pytorch. Builds the
            classifier and attaches it to it.

        Arguments:
            hidden_units: The number of hidden unites the classifier will have.
        Outputs:
            model: Variable containing the pretrained network with the classifer
                to predict.
            input_size: The input size that the classifier will have. Will be
                used in the save_checkpoint function.
        '''
        # Download models
        alexnet = models.alexnet(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        # Create Model dictionary
        model_dic = {'alexnet': alexnet, 'vgg16': vgg16}
        model_type = self.model_type

        output_size = 102
        dropout_rate = 0.5

        # Check the model name to determine how model is built.
        if model_type == 'alexnet':
            model = models[model_type]
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

            return model, input_size

        elif model_type == 'vgg16':
            model = models[model_type]
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

            return model, input_size

    def train_network(self, model, train_data, validation_data, learnrate,
        epochs, gpu_on):
        '''
        Builds a network using feedforward and backpropagation with the VGG16
            pretrained model.

        Arguments:
            model: The VGG16 pretrained model.
            train_data: The training data set used to train the model.
            validation_data: The validation data used to test for overfitting
                while training.
            learnrate: The learnate the model will use to adjust weights on the
                backpropagation.
            epochs: The amount of epochs the model will run.
            gpu_on: Boolean value indicating if user wants to use the gpu to
                train.

        Outputs:
            Trained neural network that has been tested with validation data to
                prevent overfitting.
            optimizer: The optimizer that the model used. It will be used in the
                save_checkpoint function.
        '''
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
        print("Train Network")
        steps = 0
        print_every = 20
        # Switch model dpending on user input.
        if gpu_on == False:
            model.to('cpu')
        else:
            model.to('cuda')
        # Loop through all epochs
        print('Training starting...')
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_data):
                steps += 1
                if gpu_on == False:
                    inputs, labels = inputs.to('cpu'), labels.to('cpu')
                else:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
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
                        test_loss, accuracy = self.validate_network(model,
                            validation_data, criterion, gpu_on)

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                        "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.4f}.. ".format(test_loss/len(validation_data)),
                        "Validation Accuracy: {:.4f}".format(accuracy/len(validation_data)))
                    # Set running loss back to 0
                    running_loss = 0
                    # Return your model to training model.
                    model.train()
        print('Training complete')
        return optimizer

    def validate_network(self, model, validation_data, criterion, gpu_on):
        '''
        Validates the the accuracy of the neural network using validation data.

        Arguments:
            model: The VGG16 pretrained model
            validation_data: The validation data used to test for overfitting
                while training
            criterion = The criterion input used for the model.
            gpu_on: Boolean value indicating if user wants to use the gpu to
                train.

        Outputs:
            The accuracy of the neural network represented in test_lost and
                accuracy.
        '''
        test_loss = 0
        accuracy = 0
        for inputs, labels in validation_data:

            if gpu_on == False:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
            else:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    def test_network(self, model, test_data, gpu_on):
        '''
        Tests the accuracy of the trained neural network using the test data

        Arguments:
            model: The model that the will be tested.
            test_data: Tests data used to test neural network.
            gpu_on: Boolean value indicating if user wants to use the gpu to
                train.
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
                if gpu_on == False:
                    images, labels = images.to('cpu'), labels.to('cpu')
                else:
                    images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' %
            (100 * correct / total))
        print('Testing complete.')

    def save_checkpoint(self, model, model_type, input_size, hidden_units,
        optimizer, class_to_idx, epochs):
        '''
        Saves a checkpoint of the network that can be loaded.

        Arguments:
            model: The model that the will be tested.
            input_size: The input size of the model's classifier.
            hidden_units: The number of hidden unites the classifier will have.
            optimizer: The optimizer that the model used.
            class_to_idx: That image dataset that will be used in inference.
            epochs: The amount of epochs the model will run.
        Outputs:
            A save file with the current state of the network saved.
        '''
        model.to('cpu')
        checkpoint = {'arch': model_type, 'hidden_layers': hidden_units,
            'classifier': model.classifier, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict, 'image_datasets': class_to_idx}

        # Save the checkpoint
        if save_dir == None:
            # Save checkpoint in same folder as train.py
            torch.save(checkpoint, 'checkpoint.pth')
        else:
            # Save checkpoint in save directory provided by user.
            torch.save(checkpoint, save_dir + '/checkpoint.pth')

        print('Checkpoint saved')

def load_checkpoint(filepath):
    '''
    Loads the checkpoint of the network.

    Arguments:
        filepath: The name of the filepath where the saved model is located.
    Outputs:
        A loaded checkpoint for the network to use.
    '''
    print("Load checkpoint")
    checkpoint = torch.load(filepath)

    # Load the pretrained model
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)

    # Update model with classifier from checkpoint
    model.classifier = checkpoint['classifier']
    # Update model with iamge datasets from checkpoint
    model.class_to_idx = checkpoint['image_datasets']
    # Update model with state_dict from checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint
