from torch import nn

class TransposeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride = 2, final_block=False):
        super(TransposeBlock, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.activation = None
        if (final_block):
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, input, final=False):
        if (final):
            x = self.transpose(input)
            x = self.activation(x)
        else:
            x = self.transpose(input)
            x = self.batchnorm(x)
            x = self.activation(x)

        return x
