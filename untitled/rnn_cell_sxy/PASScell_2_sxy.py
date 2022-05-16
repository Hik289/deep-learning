import torch
from torch import nn

class PASScell_2(torch.nn.Module):
    """
    A simple PASS cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(PASScell_2, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_f_u1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_f_h1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_f_y1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_f = nn.Sigmoid()

        self.linear_i_z2 = nn.Linear(self.input_length+ self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_z = nn.Sigmoid()


        self.final_activation = nn.ReLU()

    def forward(self, x, o, h):

        z = torch.cat([h, x],axis = 1)

        z_temp = self.linear_i_z2(z)
        o = self.sigmoid_z(z_temp)

        x_temp = self.linear_f_u1(x)
        h_temp = self.linear_f_h1(h)
        o_temp = self.linear_f_y1(o)
        f = self.sigmoid_f(x_temp + h_temp + o_temp)

        return o, f

