import torch.nn as nn
import torch
import torch.nn.utils.weight_norm as weight_norm

class ConvNet_WN_DEEP(nn.Module):
    def __init__(self, settings):
        super(ConvNet_WN_DEEP, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.fc_depth = settings.fc_depth
        self.fc_width = settings.fc_width

        self.conv1 = weight_norm(nn.Conv2d(self.num_input_channels, self.width, kernel_size=2), dim=None)
        self.conv2 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        self.conv3 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        #self.conv4 = weight_norm(nn.Conv2d(self.width, self.width, kernel_size=2), dim=None)
        conv_layers = [self.conv1, self.conv2, self.conv3]

        fc_layers = [weight_norm(nn.Linear(self.width * 25 * 25, self.fc_width))]
        fc_layers += [weight_norm(nn.Linear(self.fc_width, self.fc_width)) for i in range(self.fc_depth)]
        fc_layers += [nn.Linear(self.fc_width, settings.num_output_classes)]
        self.fc_layers = nn.ModuleList(fc_layers)

        self.all_layers = conv_layers + fc_layers

    def forward(self, x):
        shapes = []
        # Pass input through first convolutional layer and ReLU activation
        shapes += [x.shape]
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
        #x = self.conv4(x)
        #shapes += [x.shape]
        #x = nn.functional.relu(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        for i,fc in enumerate(self.fc_layers):
            x = fc(x)
            shapes += [x.shape]
            if i != len(self.fc_layers)-1:
                x = nn.functional.relu(x)

        # Pass through fully connected layer and return the output
        #x = self.fc(x)
        return x, shapes


def convnet_wn_deep(settings):
    return ConvNet_WN_DEEP(settings)