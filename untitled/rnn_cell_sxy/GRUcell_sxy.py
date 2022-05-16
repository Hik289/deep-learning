import torch
from torch import nn

class GRUcell(torch.nn.Module):
    """
    A simple GRU cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(GRUcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_r_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_r_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_r = nn.Sigmoid()

        self.linear_z_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_z_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_z = nn.Sigmoid()

        self.linear_n_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_n_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_n = nn.Tanh()


    def forward(self, x, h):

        x_temp = self.linear_r_w1(x)
        h_temp = self.linear_r_r1(h)
        r = self.sigmoid_r(x_temp + h_temp)

        x_temp = self.linear_z_w2(x)
        h_temp = self.linear_z_r2(h)
        z =  self.sigmoid_z(x_temp + h_temp)

        x_temp = self.linear_n_w3(x)
        h_temp = self.linear_n_r3(h)
        n = self.activation_n(x_temp + r*h_temp)

        h = (1-z)*n + z*h_temp

        return n, h