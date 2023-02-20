# -*- coding:UTF-8 -*-
import numpy as np
import torch
import logging
from torch import nn
from collections import OrderedDict
from copy import deepcopy
from New_Dataloader import Size

# Define LSTM Neural Networks
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.device = device
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).to(self.device)
        # 注意oupt_put乘的数字与size的第三个数字相关
        self.fc = nn.Linear(hidden_size, output_size*Size[-1]).to(self.device)

    def forward(self, x):
        # logging.info("shape x_shape_in_lstm = {}".format(x.shape))
        # 2    8     7
        setsz, seq_n, seq_len = x.size()
        # reshape x to match the input size
        if x.size(-1) != self.input_size:
            x = x.unsqueeze(-1)
            # print("暂时替代")
        if len(x.shape) > 3:
            x = torch.reshape(x, (-1, seq_len, self.input_size))
        # type change
        x = x.float()

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)    
        out = self.fc(h_out)

        # reshape out and change type to match label
        # out = torch.reshape(out, (setsz, seq_n))
        out = torch.reshape(out, (setsz, setsz, seq_len))
        out = out.double()

        return out

def main():
    pass

if __name__ == '__main__':
    main()