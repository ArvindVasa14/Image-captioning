import torch
import numpy as np
from torch import nn

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(Decoder, self).__init__()
        self.embed_size= embed_size
        self.hidden_size= hidden_size
        self.num_layers= num_layers
        self.vocab_size= vocab_size

        # input : batch, 256
        self.lstm= nn.LSTM(self.embed_size, self.hidden_size, num_layers= self.num_layers, batch_first= True)
        self.fc= nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions):
        output, hidden= self.lstm(features)

