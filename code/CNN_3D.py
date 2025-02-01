import torch
import torch.nn as nn

class Simple3DRegressionCNN(nn.Module):
    def __init__(self):
        super(Simple3DRegressionCNN, self).__init__()

        # First group of layers: Convolution, Batch Normalization, ReLU activation function, and Max Pooling
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) 
        )

        # Second group of layers: Convolution, Batch Normalization, ReLU activation function, and Max Pooling
        self.group2 = nn.Sequential(
            nn.Conv3d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        # Calculate the size of the output tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 109, 145, 145)  # Create a dummy input tensor
            dummy_output = self.group2(self.group1(dummy_input))  # Pass the dummy input through the first two groups of layers
            output_size = dummy_output.view(-1).size(0)

        # First fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(output_size, 120),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Second fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(84, 1)
        )

    # Define the forward pass
    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = out.view(out.size(0), -1)  # Flatten the output tensor
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        out = out * 100  # Scale the output
        return out