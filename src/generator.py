from torch import nn

import sys
sys.path.insert(1, '/components')

from components.transposeblock import TransposeBlock

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.trans_block1 = TransposeBlock(self.noise_dim, 256, (3, 3))
        self.trans_block2 = TransposeBlock(256, 128, (4, 4), 1)
        self.trans_block3 = TransposeBlock(128, 64, (3, 3))
        self.trans_block4 = TransposeBlock(64, 1, (4, 4), final_block=True)

    def forward(self, input):
        x = input.view(-1, self.noise_dim, 1, 1)
        x = self.trans_block1(x)
        x = self.trans_block2(x)
        x = self.trans_block3(x)
        x = self.trans_block4(x, True)

        return x