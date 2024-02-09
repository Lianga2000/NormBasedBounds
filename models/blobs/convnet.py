
import torch.nn as nn
import torch
import torch.nn.utils.weight_norm as weight_norm

class ConvNet(nn.Module):
    def __init__(self, settings):
        super(ConvNet, self).__init__()
        
        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        
        self.conv1 = nn.Conv2d(self.num_input_channels, self.width, kernel_size=2)
        self.conv2 = nn.Conv2d(self.width, self.width, kernel_size=2)
        self.conv3 = nn.Conv2d(self.width, self.width, kernel_size=2)
        self.conv4 = nn.Conv2d(self.width, self.width, kernel_size=2)
        self.conv5 = nn.Conv2d(self.width, self.width, kernel_size=2)
        self.conv6 = nn.Conv2d(self.width, self.width, kernel_size=2)
        
        layers = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5, self.conv6]
        self.layers = nn.Sequential(*layers)
        
        self.fc = nn.Linear(self.width*22*22, settings.num_output_classes)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, settings.num_output_classes)
        #self.linera_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        # Pass input through first convolutional layer and ReLU activation
        x = self.conv1(x)
        x = nn.functional.relu(x)
        
        # Pass through the remaining four convolutional layers and ReLU activation
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layers and return the output
        x = self.fc(x)
        #x = nn.functional.relu(x)
        #x = self.fc2(x)
        #x = nn.functional.relu(x)
        #x = self.fc3(x)
        return x

def convnet(settings):
    return ConvNet(settings)
