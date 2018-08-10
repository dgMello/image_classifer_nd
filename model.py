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
        output_size = 102
        dropout_rate = 0.5

        # Check the model name to determine how model is built.
        if self.model_type == 'alexnet':
            print('Model is', self.model_type)
            model = models[self.model_type]
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

        elif self.model_type == 'vgg16':
            print('Model is', self.model_type)
            model = models[self.model_type]
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

    def train_network(self, model, train_data, validation_data, learnrate, epochs, gpu_on):
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
                        test_loss, accuracy = self.validate_network(model, validation_data, criterion, gpu_on)

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.4f}.. ".format(test_loss/len(validation_data)),
                          "Test Accuracy: {:.4f}".format(accuracy/len(validation_data)))
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
            validation_data: The validation data used to test for overfitting while training
            criterion = The criterion input used for the model.

        Outputs:
            The accuracy of the neural network represented in test_lost and accuracy.
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
                if gpu_on == False:
                    images, labels = images.to('cpu'), labels.to('cpu')
                else:
                    images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('Testing complete.')

    def save_checkpoint(self, model, input_size, hidden_units, optimizer, class_to_idx, epochs):
        print("Save checkpoint")
        checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_layers': hidden_units,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict,
              'image_datasets': class_to_idx,
              'epochs': epochs}
        # Save the checkpoint
        torch.save(checkpoint, 'checkpoint.pth')

    def load_checkpoint(self, filepath):
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

    def predict(self):
        print("Predict!")
