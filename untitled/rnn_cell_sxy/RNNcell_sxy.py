import torch
from torch import nn

class RNNcell(torch.nn.Module):
    """
    A simple RNN cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(RNNcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_rnn_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_rnn_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_rnn = nn.Sigmoid()

        self.activation_final = nn.Tanh()



    def forward(self, x, h):

        x_temp = self.linear_rnn_w1(x)
        h_temp = self.linear_rnn_r1(h)
        h = self.sigmoid_rnn(x_temp + h_temp)

        return h, h