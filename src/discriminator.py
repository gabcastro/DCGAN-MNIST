from torch import nn

import sys
sys.path.insert(1, '/components')

from components.cnnblock import CNNBlock

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.cnn_block1 = CNNBlock(1, 16, (3, 3))
        self.cnn_block2 = CNNBlock(16, 32, (5, 5))
        self.cnn_block3 = CNNBlock(32, 64, (5, 5))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=1)

    def forward(self, input):
        x = self.cnn_block1(input)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)

        x = self.flatten(x)
        o = self.linear(x)

        return o