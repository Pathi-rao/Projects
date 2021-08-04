from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, BatchNorm2d, Dropout, MaxPool2d
from torch.nn.modules import batchnorm
from torchsummary import summary


class CNN(Module):
    
    def __init__(self):
        super(CNN,self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 512, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(512),
            ReLU(inplace=True),

            # ((layer_size - kernel_size + 2 * padding) / stride) + 1
            # padding_default = 0, stride_default = 1

            #MaxPool2d(kernel_size=2, stride=2),
            Conv2d(512, 256, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(256),
            ReLU(inplace=True),

            Conv2d(256, 128, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(128),
            ReLU(),

            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = Sequential(
            Linear(128*2*2, 512),
            Dropout(0.2, inplace= True),

            Linear(512, 256),
            Dropout(0.2, inplace= True),

            Linear(256, 128),
            Dropout(0.2, inplace= True),

            Linear(128, 64),
            Dropout(0.2, inplace= True),

            Linear(64, 7) #7 classes
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = CNN()

summary(model, (1, 48, 48))