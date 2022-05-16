import torch
from torch import nn

class PASScell_1(torch.nn.Module):
    """
    A simple PASS cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(PASScell_1, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_f_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_f_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_f = nn.Sigmoid()

        self.linear_i_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_i_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_i = nn.Sigmoid()

        self.linear_k_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_k_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_k = nn.Sigmoid()

        self.linear_g_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_g_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_g = nn.Sigmoid()

        self.final_activation = nn.ReLU()

    def forward(self, x, o, h):

        x_temp = self.linear_f_w1(x)
        h_temp = self.linear_f_r1(h)
        f = self.sigmoid_f(x_temp + h_temp)

        x_temp = self.linear_i_w2(x)
        h_temp = self.linear_i_r2(h)
        i =  self.sigmoid_i(x_temp + h_temp)

        x_temp = self.linear_k_w3(x)
        h_temp = self.linear_k_r3(h)
        k = self.sigmoid_k(x_temp + h_temp)

        x_temp = self.linear_g_w4(x)
        h_temp = self.linear_g_r4(h)
        g = self.sigmoid_g(x_temp + h_temp)

        o = f * o - i* self.final_activation(g)
        h = k * self.final_activation(o)

        return o, h

