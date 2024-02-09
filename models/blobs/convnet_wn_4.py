import torch.nn as nn
import torch
import torch.nn.utils.weight_norm as weight_norm


class ConvNet_WN_4(nn.Module):
    def __init__(self, settings):
        super(ConvNet_WN_4, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width

        self.conv1 = weight_norm(nn.Conv2d(self.num_input_channels, self.width, kernel_size=2), dim=None)
        self.conv2 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        self.conv3 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        self.conv4 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        # self.conv5 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)

        layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(self.width * 24 * 24, settings.num_output_classes)
        
        self.all_layers = [self.conv1, self.conv2, self.conv3, self.fc]

    def forward(self, x):
        shapes = []
        shapes += [x.shape]
        # Pass input through first convolutional layer and ReLU activation
        x = self.conv1(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)

        # Pass through the remaining four convolutional layers and ReLU activation
        x = self.conv2(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.conv3(x)
        shapes += [x.shape]
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Pass through fully connected layer and return the output
        x = self.fc(x)
        return x, shapes


def convnet_wn_4(settings):
    return ConvNet_WN_4(settings)
    