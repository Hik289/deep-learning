import torch
from torch import nn

class bottleneck_1(nn.Module):
    def __init__(self,input_dim=3, output_dim = 6):
        super(bottleneck_1,self).__init__()
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(input_dim*2)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(input_dim*2)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=input_dim*2, out_channels=output_dim, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(output_dim)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1),
                                      nn.BatchNorm1d(output_dim))

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)
    
        shortcut = self.shortcut(x)

        x_signal = x_signal + shortcut[:,:,1:-1]

        out = self.act(x_signal)

        return out


class bottleneck_2(nn.Module):
    def __init__(self,input_dim=3, output_dim = 6):
        super(bottleneck_2,self).__init__()
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(input_dim*2)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(input_dim*2)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(input_dim)
        self.act3 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        x_signal = x_signal + (x)[:,:,1:-1]

        out = self.act(x_signal)

        return out