from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride = 2):
        super(CNNBlock, self).__init__()

        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv2d(input)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x
