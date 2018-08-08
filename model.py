%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__

class Model:
    def __init__(self):
        print("Model Created")

    def train_network(model, train_data, validation_data, epochs, device):
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
        # Switch model to cuda to increate spead of training
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
