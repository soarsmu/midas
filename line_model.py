import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F

HIDDEN_DIM_DROPOUT_PROB = 0.3


class PatchClassifierByLine(nn.Module):
    def __init__(self):
        super(PatchClassifierByLine, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, embedding_batch):
        self.lstm.flatten_parameters()
        _, (embedding_final_hidden_state, _) = self.lstm(embedding_batch)
        x = embedding_final_hidden_state[0]

        x = self.relu(x)

        x = self.drop_out(x)

        out = self.out_proj(x)

        return out
