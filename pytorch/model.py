# TODO automatic Liner layer calculations

from torch import nn
from hyperparameters import KERNEL_SIZE, STRIDE, OUTPUT_CLASSES, KERNEL_NUMBER

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, KERNEL_NUMBER, kernel_size=KERNEL_SIZE, padding=0, stride=STRIDE),
            # max(0,x) all negative become 0
            nn.ReLU(),
            nn.MaxPool2d(2),
            # # input channels for the second conv layer = output channel of the previous
            nn.Conv2d(KERNEL_NUMBER, KERNEL_NUMBER*2, kernel_size=KERNEL_SIZE, padding=0, stride=STRIDE),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # input features for fc layer
        # after Conv2D: (28 - 3 + 0 )/ 1 + 1 = 26 (input_size - kernel_size + 2*padding)/stride + 1
        # after MaxPool: 26/2 = 13
        # Features = 13x13x32 = 5408
        self.fc_layer_block = nn.Sequential(
            nn.Flatten(),
            # Linear, 128 the power of two, not computationally expensive, additional depth, fast training
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, OUTPUT_CLASSES)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.fc_layer_block(x)
        return x