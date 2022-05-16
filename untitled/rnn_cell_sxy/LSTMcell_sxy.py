import torch
from torch import nn

class LSTMcell(torch.nn.Module):
    """
    A simple LSTM cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(LSTMcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_gate = nn.Tanh()

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()



    def forward(self, x, c, h):

        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)

        # Equation 2. forget gate
        x_temp = self.linear_forget_w1(x)
        h_temp = self.linear_forget_r1(h)
        f =  self.sigmoid_forget(x_temp + h_temp)

        # Equation 3. updating the cell memory
        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        k = self.activation_gate(x_temp + h_temp)
        g = k * i
        c = f * c
        c = g + c

        # Equation 4. calculate the main output gate
        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
        o =  self.sigmoid_hidden_out(x_temp + h_temp)

        # Equation 5. produce next hidden output
        h = o * self.activation_final(c)

        return o, c, h