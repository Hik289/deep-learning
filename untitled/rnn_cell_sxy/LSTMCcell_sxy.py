import torch
from torch import nn

class LSTMCcell(torch.nn.Module):
    """
    A simple LSTMC cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(LSTMCcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_gate_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_1 = nn.Sigmoid()

        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_2 = nn.Sigmoid()

        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_3 = nn.Tanh()

        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_gate_c4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_gate_4 = nn.Sigmoid()

        self.activation_final = nn.ReLU()

    def forward(self, x, c, h):

        x_temp = self.linear_gate_w1(x)
        h_temp = self.linear_gate_r1(h)
        c_temp = self.linear_gate_c1(c)
        i = self.sigmoid_gate_1(x_temp + h_temp + c_temp)

        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        c_temp = self.linear_gate_c2(c)
        f = self.sigmoid_gate_2(x_temp + h_temp + c_temp)

        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        c_temp = self.linear_gate_c3(c)
        k = self.sigmoid_gate_3(x_temp + h_temp + c_temp)
        c = f*c + i* self.activation_final(k)

        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
        c_temp = self.linear_gate_c4(c)
        o = self.sigmoid_gate_4(x_temp + h_temp + c_temp)

        h = o * self.activation_final(c)

        return o, c, h