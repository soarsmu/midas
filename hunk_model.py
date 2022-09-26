import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F

HIDDEN_DIM_DROPOUT_PROB = 0.3


class PatchClassifierByHunk(nn.Module):
    def __init__(self):
        super(PatchClassifierByHunk, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(4 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch):
        self.lstm.flatten_parameters()
        before_out, (before_final_hidden_state, _) = self.lstm(before_batch)
        before_vector = before_out[:, 0]

        after_out, (after_final_hidden_state, _) = self.lstm(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)

        x = self.drop_out(x)

        out = self.out_proj(x)

        return out
